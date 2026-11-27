"""Stage 3 of Run A: VideoJev picks ONE final answer from the drafted captions.

For every question, the drafter's captions (candidates.jsonl from draft_videosiah.py) are
de-duplicated and given to VideoJev-4B's single-choice QA head as options ("A. <caption>",
"B. <caption>", ... -- the lettered format of the VideoJev README / training data). VideoJev
reads the WHOLE video (not just the grounded clip) with its QA protocol and scores every option
directly (softmax over options, no text generation); the top-scoring caption is the final answer.

With --video-features (the default in run_textdecoder_sampling.sh), VideoJev reuses the
whole-video features cached by the grounding stage, so selection is a single choice forward pass
per video with no decoding. Those features follow the GROUNDING sampling (4 FPS, <= 2048 frames):
identical to the QA protocol's frames for videos > ~17 min (both hit the 2048-frame cap), slightly
denser for shorter ones. Videos without a cache fall back to decoding with the QA protocol
(2 FPS, <= 2048 frames, context 98304 instead of the official 262144, which does not fit 24 GB).
Each record says which path was used (`feature_source`).

Outputs, in --out:
  predictions.jsonl   one final answer per question; `prediction` is what the LLM judge grades
                      (scripts/judge_videosiah_longvt.py --predictions predictions.jsonl)
  final.md / final.csv  manual-inspection sheet: question, GT, grounded interval, final answer
                      (+ every option's VideoJev probability, collapsed)

Run under torchrun for multi-GPU sharding (whole videos per GPU); resumes by id.
"""
import argparse
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import string
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from videojev_runtime import make_prepare_video, patch_decode_cap, run_with_context_fallback  # noqa: E402


def mmss(seconds):
    seconds = int(round(seconds))
    return f'{seconds // 60:d}:{seconds % 60:02d}'


def write_sheet(records, grounding, out):
    md = [f'# VideoJev final answers ({len(records)} questions)\n',
          'One final answer per question: the caption VideoJev\'s QA head scored highest among the '
          'drafted captions. Intervals are VideoJev grounding predictions (NOT ground truth).\n']
    rows = []
    for n, r in enumerate(records, 1):
        g = grounding.get(r['id'], {})
        interval = g.get('interval')
        where = f'{mmss(interval[0])}–{mmss(interval[1])} ({interval[0]:.0f}–{interval[1]:.0f}s)' if interval else 'n/a'
        md += [f'\n---\n\n## {n}. `{r["id"]}` — `{Path(r["video_path"]).name}`\n',
               f'**Q:** {r["question"]}\n', f'**GT:** {r["answer"]}\n',
               f'**Grounded clip:** {where}\n',
               f'**Final answer** (option {r["prediction_letter"]}, p = {r["prediction_prob"]:.3f}, '
               f'chosen from {len(r["options"])} unique captions):\n',
               f'> {r["prediction"]}\n',
               '<details><summary>all options (VideoJev probability)</summary>\n',
               '| option | p | caption | drafted as |', '|---|---|---|---|']
        for letter, text, prob, kinds in zip(string.ascii_uppercase, r['options'], r['probabilities'], r['option_sources']):
            mark = ' ✅' if text == r['prediction'] else ''
            md.append(f'| {letter}{mark} | {prob:.3f} | {text} | {", ".join(kinds)} |')
        md.append('\n</details>')
        rows.append(dict(id=r['id'], video=Path(r['video_path']).name, question=r['question'], answer=r['answer'],
                         interval=where, final_answer=r['prediction'], option=r['prediction_letter'],
                         probability=round(r['prediction_prob'], 4), num_options=len(r['options']),
                         greedy_caption=r['greedy_caption'], correct=''))
    (out / 'final.md').write_text('\n'.join(md) + '\n')
    with (out / 'final.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--candidates', required=True, help='candidates.jsonl from draft_videosiah.py')
    parser.add_argument('--grounding', help='grounding.jsonl (only to show the interval in final.md)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint', default='ZhuYuhan/VideoJev-4B')
    parser.add_argument('--fps', type=float, default=2.0, help='QA fps (official: 2)')
    parser.add_argument('--max-frames', type=int, default=2048, help='QA frame cap (official: 2048)')
    parser.add_argument('--context', type=int, default=98304,
                        help='QA context tokens (official 262144 does not fit 24 GB; shrinks on OOM)')
    parser.add_argument('--min-context', type=int, default=32768)
    parser.add_argument('--decode-max-edge', type=int, default=1280,
                        help='Downscale frames at decode time (0 = official full-res decode)')
    parser.add_argument('--video-features', help='Whole-video feature cache written by '
                        'scripts/ground_videosiah.py --video-features; videos found there skip decoding entirely')
    parser.add_argument('--max-pixels', type=int, default=640 * 640,
                        help='Per-frame pixel ceiling when decoding (match grounding\'s --max-pixels; '
                             'cached videos use grounding\'s features as-is)')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    import av
    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoModel, AutoProcessor

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('gloo', timeout=timedelta(hours=12))

    source = Path(args.questions)
    order = [r['id'] for r in json.loads(source.read_text())]
    paths = {r['id']: str((source.parent / r['video_path']).resolve()) for r in json.loads(source.read_text())}
    drafts = {json.loads(l)['id']: json.loads(l) for l in Path(args.candidates).read_text().splitlines() if l.strip()}
    ids = [i for i in order if i in drafts][:args.limit] if args.limit else [i for i in order if i in drafts]
    grounding = ({json.loads(l)['id']: json.loads(l) for l in Path(args.grounding).read_text().splitlines() if l.strip()}
                 if args.grounding else {})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f'predictions_rank{rank}.jsonl'
    done = {json.loads(l)['id'] for l in shard.read_text().splitlines()} if shard.exists() else set()
    videos = sorted({paths[i] for i in ids})
    owned = set(videos[rank::world])
    todo = {}
    for i in ids:
        if paths[i] in owned and i not in done:
            todo.setdefault(paths[i], []).append(i)

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
                                      device_map={'': local}, attn_implementation='flash_attention_2')
    model.prepare_for_inference()
    patch_decode_cap(model, args.decode_max_edge)
    prepare_video = make_prepare_video(model, args.max_pixels)

    def options_for(draft):
        """De-duplicate captions (exact text after strip), keeping first-seen order and provenance."""
        options, provenance = [], []
        for text, src in zip(draft['candidates'], draft.get('sources') or [{}] * len(draft['candidates'])):
            text = ' '.join(text.split())
            label = f'{src.get("kind", "?")}'
            if not text or '<|' in text:
                continue
            if text in options:
                provenance[options.index(text)].append(label)
            else:
                options.append(text)
                provenance.append([label])
        return options, provenance

    with shard.open('a') as stream, torch.inference_mode():
        for path in tqdm(sorted(todo), desc=f'GPU {rank}', unit='video', position=rank, dynamic_ncols=True):
            batch = []
            for qid in todo[path]:
                options, provenance = options_for(drafts[qid])
                batch.append((qid, options, provenance))
            single = [b for b in batch if len(b[1]) < 2]  # one unique caption: nothing to choose
            multi = [b for b in batch if len(b[1]) >= 2]
            questions = [dict(id=qid, task='choice', query=drafts[qid]['question'],
                              options=[f'{string.ascii_uppercase[k]}. {o}' for k, o in enumerate(options)])
                         for qid, options, _ in multi]

            cache = Path(args.video_features) / f'{Path(path).stem}.pt' if args.video_features else None

            def score(context):
                pixels, record = prepare_video(processor, path, fps=args.fps, max_frames=args.max_frames,
                                               context=context)
                features = model.encode_video(pixels, torch.tensor([record['grid_thw']]))
                del pixels
                results = model.predict_features(processor, features, record, questions, context=context) \
                    if questions else []
                return results, record
            try:
                if cache is not None and cache.exists():
                    # Reuse grounding's whole-video features: no decode / resize / vision encode, only the
                    # one shared choice pass. Run at the context the features were built for, so the
                    # video+questions packing matches how the features were sized.
                    saved = torch.load(cache, map_location='cpu')
                    record, context = saved['record'], saved['context']
                    features = saved['features'].to(model.device)
                    results = model.predict_features(processor, features, record, questions, context=context) \
                        if questions else []
                    del features, saved
                    source_info = dict(feature_source='grounding_cache', qa_fps=record['fps'],
                                       sampling_fps=None, cache=str(cache))
                else:
                    if cache is not None:
                        tqdm.write(f'GPU {rank}: no cached features for {Path(path).name}; decoding the video')
                    (results, record), context = run_with_context_fallback(
                        score, args.context, args.min_context,
                        log=lambda m: tqdm.write(f'GPU {rank}: {Path(path).name}: {m}'))
                    source_info = dict(feature_source='decoded', qa_fps=args.fps, sampling_fps=args.fps)
            except (av.error.FFmpegError, ValueError, RuntimeError) as error:
                torch.cuda.empty_cache()
                with (out / f'skipped_rank{rank}.jsonl').open('a') as log:
                    log.write(json.dumps(dict(video_path=path, questions=todo[path], error=str(error))) + '\n')
                tqdm.write(f'GPU {rank}: skipped {path}: {error}')
                continue
            by_id = {r['id']: r for r in results}
            for qid, options, provenance in batch:
                draft = drafts[qid]
                if len(options) >= 2:
                    probs = [float(p) for p in by_id[qid]['probabilities'][:len(options)]]
                    pick = max(range(len(options)), key=probs.__getitem__)
                else:
                    probs, pick = [1.0] * len(options), 0
                stream.write(json.dumps(dict(
                    id=qid, video_path=path, question=draft['question'], answer=draft['answer'],
                    prediction=options[pick], prediction_letter=string.ascii_uppercase[pick],
                    prediction_prob=probs[pick], options=options, probabilities=probs,
                    option_sources=provenance, greedy_caption=' '.join(draft['candidates'][0].split()),
                    selector=args.checkpoint, **source_info, max_frames=args.max_frames, context=context,
                    frames=record['frames'], grid_thw=record['grid_thw'],
                    drafter=draft.get('model'), draft_mode=draft.get('mode')), ensure_ascii=False) + '\n')
                stream.flush()
            torch.cuda.empty_cache()

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {}
        for f in sorted(out.glob('predictions_rank*.jsonl')):
            for line in f.read_text().splitlines():
                record = json.loads(line)
                merged[record['id']] = record
        records = [merged[i] for i in ids if i in merged]
        (out / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        if records:
            write_sheet(records, grounding, out)
        print(json.dumps(dict(selected=len(records), missing=len(ids) - len(records),
                              predictions=str(out / 'predictions.jsonl'), sheet=str(out / 'final.md'))))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
