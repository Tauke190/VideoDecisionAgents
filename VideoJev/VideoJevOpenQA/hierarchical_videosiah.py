"""Uncertainty-aware hierarchical sampling for open-ended VideoSIAH QA (no candidate generation).

System 1 -- VideoJev-4B grounding, coarse to fine, ONE peak per level:
  level 0: --frames frames uniformly over the whole video -> per-cell logits (1 cell = 2 frames).
  Kadane (official max-likelihood segment, threshold --threshold) picks the peak segment; its
  window = segment widened by --margin-cells cells on each side.
  confidence = share of the total membership probability sigmoid(logit) inside that window.
  confident  = confidence >= --conf-threshold AND peak probability >= --min-peak.
  If not confident, re-ground on the window only (--frames frames inside it, <= --max-fps), until
  confident, --max-depth levels, cells <= --min-cell seconds, or the window stops shrinking.
  Never confident -> use the window of the most confident level (flagged confident=false).
  --peaks K > 1: at the first uncertain level, also zoom into the next K-1 peaks (iterated Kadane,
  peak probability >= --peak-min); each branch then zooms on one peak. Branches are ranked by
  prior x confidence, prior = that peak's probability share at the level where they split.
System 2 -- Qwen3.5-4B answers directly (greedy, short answer) from --answer-frames frames sampled
  uniformly inside the final window at up to --answer-max-pixels per frame, with real timestamps.

Frames are read by seeking (only the needed frames are decoded, never the whole video).

Outputs in --out: predictions.jsonl (`prediction` is what the judge grades; every level's
probabilities, windows and confidences are stored), inspect.md (per-question walk-through with a
probability sparkline per level), summary.json. Resumes by id; torchrun shards questions over GPUs.
"""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from videojev_runtime import videojev_modules  # noqa: E402

PROMPTS = {
    'phrase': '{question}\nAnswer the question directly with a short answer or phrase.',
    'short': ('{question}\nGive a short answer in one brief sentence. Be specific: include the key details '
              'the question asks about, such as names, colors, numbers or objects.'),
    # CG-Bench multiple choice (option format as in VLMEvalKit)
    'mcq': 'Question: {question}\n{options}\nAnswer with the option\'s letter from the given choices directly.',
}
LETTERS = 'ABCDEFGH'


def parse_letter(text, n):
    """First standalone option letter in the reply, or None."""
    import re
    m = re.search(rf'\b([{LETTERS[:n]}])\b', text or '')
    return m.group(1) if m else None
BARS = '▁▂▃▄▅▆▇█'


def mmss(seconds):
    seconds = int(round(seconds))
    return f'{seconds // 60:d}:{seconds % 60:02d}'


def video_duration(path):
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base) if stream.duration is not None else container.duration / 1e6


def read_frames(path, times, max_edge):
    """RGB frames nearest to each (increasing) time in seconds, by seeking; one fixed size per call."""
    import av
    import numpy as np
    frames, actual, size = [], [], None

    def rgb(frame):
        nonlocal size
        if size is None:
            scale = min(1., max_edge / max(frame.width, frame.height)) if max_edge else 1.
            size = (max(2, round(frame.width * scale / 2) * 2), max(2, round(frame.height * scale / 2) * 2))
        return frame.to_ndarray(format='rgb24', width=size[0], height=size[1])

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'  # 'AUTO' (frame threading) deadlocks on some HEVC zoom decodes
        base = float(stream.time_base)
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.
        decoder, previous, current = None, None, None
        for target in times:
            if decoder is None or current is None or target > current[1] + 2.0:
                container.seek(max(0, int((target + origin) / base)), stream=stream, backward=True, any_frame=False)
                decoder, previous, current = container.decode(stream), None, None
            while current is None or current[1] < target:
                try:
                    frame = next(decoder)
                except StopIteration:
                    break
                if frame.pts is None:
                    continue
                previous, current = current, (frame, float(frame.pts * frame.time_base) - origin)
            options = [x for x in (previous, current) if x is not None]
            if not options:
                raise ValueError(f'no frame decodable near {target:.2f}s in {path}')
            frame, stamp = min(options, key=lambda x: abs(x[1] - target))
            frames.append(rgb(frame))
            actual.append(stamp)
    return np.stack(frames), actual


def kadane(z):
    best = current = float(z[0])
    start = best_start = best_end = 0
    for i in range(1, len(z)):
        if current <= 0:
            start, current = i, float(z[i])
        else:
            current += float(z[i])
        if current > best:
            best, best_start, best_end = current, start, i
    return best_start, best_end


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', default='data/benchmarks/videosiah/questions.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids', help='Comma-separated question ids to run instead of the first --limit')
    parser.add_argument('--checkpoint', default='ZhuYuhan/VideoJev-4B')
    parser.add_argument('--answer-model', default='Qwen/Qwen3.5-4B')
    # System 1
    parser.add_argument('--frames', type=int, default=64, help='Frames per grounding level (even)')
    parser.add_argument('--max-fps', type=float, default=4.0, help='Sampling-rate ceiling inside short windows')
    parser.add_argument('--grounding-max-pixels', type=int, default=640 * 640, help='Per-frame cap (official 640*640)')
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--threshold', type=float, default=0.25, help='Kadane threshold (official 0.25)')
    parser.add_argument('--margin-cells', type=int, default=1)
    parser.add_argument('--conf-threshold', type=float, default=0.8)
    parser.add_argument('--min-peak', type=float, default=0.25, help='Peak probability also required to be confident')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum grounding levels (level 0 included)')
    parser.add_argument('--min-cell', type=float, default=2.0, help='Stop zooming once cells are this short (s)')
    parser.add_argument('--peaks', type=int, default=1,
                        help='Children per uncertain node: its top peaks, each zoomed separately (1 = a single chain)')
    parser.add_argument('--peak-min', type=float, default=0.1, help='Probability an extra peak needs to become a child')
    parser.add_argument('--max-nodes', type=int, default=10, help='Grounding calls per question (tree size budget)')
    # System 2
    parser.add_argument('--answer-frames', type=int, default=16)
    parser.add_argument('--answer-max-pixels', type=int, default=640 * 640)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--prompt', choices=['phrase', 'short', 'mcq'], default='short',
                        help='mcq = multiple choice from the row\'s `choices` (CG-Bench)')
    parser.add_argument('--clips', action='store_true',
                        help='Give each leaf to Qwen as its own labelled clip (start-end) instead of one joined video')
    parser.add_argument('--pad', type=float, default=1.0, help='Widen the final window by this factor around its centre')
    parser.add_argument('--top-leaves', type=int, default=2,
                        help='Qwen answers from the top-K ranked leaves (score = path prior x leaf confidence)')
    parser.add_argument('--fill-coarse', action='store_true',
                        help='If fewer than K leaves, fill with extra peaks of the coarse level (old merge behaviour)')
    parser.add_argument('--ground-only', action='store_true', help='Run System 1 only (no Qwen answer), e.g. for CG-Bench')
    parser.add_argument('--sample', type=int, help='Random sample of N questions (seeded) instead of the first --limit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reuse', help='predictions.jsonl of an earlier run: reuse its grounding, run System 2 only')
    parser.add_argument('--decode-max-edge', type=int, default=1280)
    args = parser.parse_args()
    if args.frames % 2 or args.frames < 4:
        parser.error('--frames must be even and >= 4')

    import numpy as np
    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    from transformers.video_utils import VideoMetadata

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('gloo', timeout=timedelta(hours=12))

    source = Path(args.questions)
    if source.suffix == '.jsonl':
        rows = [json.loads(l) for l in source.read_text().splitlines()]
        for r in rows:
            r.setdefault('question', r.get('query'))
            r.setdefault('answer', '')
    else:
        rows = json.loads(source.read_text())
    if args.sample:
        import random
        rows = sorted(random.Random(args.seed).sample(rows, args.sample), key=lambda r: r['id'])
    if args.ids:
        wanted = args.ids.split(',')
        rows = [r for r in rows if r['id'] in wanted]
    elif args.limit:
        rows = rows[:args.limit]
    for row in rows:
        row['video_path'] = str((source.parent / row['video_path']).resolve())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f'predictions_rank{rank}.jsonl'
    done = {json.loads(l)['id'] for l in shard.read_text().splitlines()} if shard.exists() else set()
    todo = [r for r in rows[rank::world] if r['id'] not in done]

    reuse = {json.loads(l)['id']: json.loads(l) for l in open(args.reuse)} if args.reuse else None
    if reuse:
        todo = [r for r in todo if r['id'] in reuse]
    else:
        vj_processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
        vj = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
                                       device_map={'': local}, attn_implementation='flash_attention_2')
        vj.prepare_for_inference()
        modules = videojev_modules(vj)
        processing, temporal = modules['processing_videojev'], modules['videojev_temporal']
    if not args.ground_only:
        qa_processor = AutoProcessor.from_pretrained(args.answer_model)
        qa = AutoModelForImageTextToText.from_pretrained(args.answer_model, dtype=torch.bfloat16,
                                                         attn_implementation='flash_attention_2',
                                                         device_map={'': local}).eval()
    shift = math.log(args.threshold / (1 - args.threshold))

    def native_record(video, length, fps):
        """== processing.prepare_video after read_video, for an arbitrary uniform clip of `length` s."""
        n, height, width, _ = video.shape
        vp = vj_processor.video_processor
        budget = args.context * 4 // 5

        def geometry(cap):
            h, w = smart_resize(n, height, width, temporal_factor=vp.temporal_patch_size,
                                factor=vp.patch_size * vp.merge_size, min_pixels=vp.size['shortest_edge'],
                                max_pixels=n * cap)
            return (n + 1) // 2 * (h // 32) * (w // 32)
        low, high = max(1024, math.ceil(vp.size['shortest_edge'] / n)), args.grounding_max_pixels
        if geometry(low) > budget:
            raise ValueError('clip cannot fit the context at minimum resolution')
        while low < high:
            mid = (low + high + 1) // 2
            low, high = (mid, high) if geometry(mid) <= budget else (low, mid - 1)
        meta = VideoMetadata(total_num_frames=n, fps=fps, frames_indices=list(range(n)))
        native = vp(videos=[video], video_metadata=[meta], do_sample_frames=False,
                    size=dict(shortest_edge=vp.size['shortest_edge'], longest_edge=n * low), return_tensors='pt')
        grid = native['video_grid_thw'][0].tolist()
        times, bounds = temporal.pair_timeline(np.arange(n, dtype=np.float64) / fps, length)
        return native['pixel_values_videos'], dict(
            frames=n, fps=fps, grid_thw=grid, tokens=int(np.prod(grid)) // 4, score_times=times.tolist(),
            score_bounds=bounds.tolist(), sampled_duration=length)

    def ground(path, start, end, query, depth):
        length = end - start
        n = max(4, min(args.frames, int(length * args.max_fps) // 2 * 2))
        fps = n / length
        clock = time.perf_counter()
        video, _ = read_frames(path, start + np.arange(n) / fps, args.decode_max_edge)
        decode_s = time.perf_counter() - clock
        clock = time.perf_counter()
        pixels, record = native_record(video, length, fps)
        features = vj.encode_video(pixels, torch.tensor([record['grid_thw']]))
        (batch, _), = list(processing.prepare_query_groups(
            vj_processor, record, [dict(id='q', task='grounding', query=query)], args.context))
        output = vj(**batch, visual_features=features)
        logits = output.logits[0, :int(output.score_mask[0].sum())].float().cpu().numpy()
        torch.cuda.synchronize()
        ground_s = time.perf_counter() - clock
        bounds = np.asarray(record['score_bounds'])
        lo, hi = kadane(logits - shift)
        _, raw = temporal.decode(logits.tolist(), record['score_times'], length, args.threshold, 1,
                                 bounds=record['score_bounds'], return_raw=True)
        if not np.allclose(raw, [bounds[lo, 0], bounds[hi, 1]]):
            raise AssertionError(f'Kadane mismatch with official decode: {raw} vs {bounds[lo, 0]}, {bounds[hi, 1]}')
        probs = 1 / (1 + np.exp(-logits.astype(np.float64)))

        def describe(lo, hi):
            wlo, whi = max(0, lo - args.margin_cells), min(len(probs) - 1, hi + args.margin_cells)
            return dict(segment=[start + bounds[lo, 0], start + bounds[hi, 1]], segment_cells=[int(lo), int(hi)],
                        window=[start + bounds[wlo, 0], start + bounds[whi, 1]], window_cells=[int(wlo), int(whi)],
                        confidence=float(probs[wlo:whi + 1].sum() / probs.sum()), peak=float(probs[lo:hi + 1].max()))

        # extra peaks: iterated Kadane outside the windows already taken
        peaks = [describe(lo, hi)]
        z = logits - shift
        while len(peaks) < args.peaks:
            z = z.copy()
            wlo, whi = peaks[-1]['window_cells']
            z[wlo:whi + 1] = -1e9
            if (z > -1e9).sum() == 0:
                break
            extra = describe(*kadane(z))
            if extra['peak'] < args.peak_min:
                break
            peaks.append(extra)
        main = peaks[0]
        return dict(depth=depth, span=[start, end], frames=n, fps=fps, cell=2 / fps, grid=record['grid_thw'],
                    **main, confident=bool(main['confidence'] >= args.conf_threshold and main['peak'] >= args.min_peak),
                    peaks=peaks[1:], probabilities=[round(float(p), 5) for p in probs],
                    decode_s=decode_s, ground_s=ground_s)

    def answer(path, windows, question, choices=None, duration=None):
        windows = sorted(windows)
        times = np.concatenate([a + (np.arange(args.answer_frames) + 0.5) * (b - a) / args.answer_frames
                                for a, b in windows])
        clock = time.perf_counter()
        frames, actual = read_frames(path, times, args.decode_max_edge)
        decode_s = time.perf_counter() - clock
        clock = time.perf_counter()
        # fps=1 with frame "indices" = seconds makes the processor print each frame group's real time.
        def meta(ts):
            return VideoMetadata(total_num_frames=len(ts), fps=1.0, frames_indices=[round(t, 2) for t in ts],
                                 duration=float(ts[-1]) + 1)
        options = '\n'.join(f'{LETTERS[i]}:{c}' for i, c in enumerate(choices or []))
        text = PROMPTS[args.prompt].format(question=question, options=options)
        if args.clips:
            # one labelled video per leaf, so the gaps between leaves are explicit
            k = args.answer_frames
            videos = [frames[i * k:(i + 1) * k] for i in range(len(windows))]
            metas = [meta(actual[i * k:(i + 1) * k]) for i in range(len(windows))]
            total = f' of a {mmss(duration)} video' if duration else ''
            content = [{'type': 'text', 'text': f'{len(windows)} clip(s){total}, in time order; '
                                                'the parts of the video between clips are not shown.\n'}]
            for i, (a, b) in enumerate(windows, 1):
                content += [{'type': 'text', 'text': f'Clip {i} ({mmss(a)}-{mmss(b)}):'}, {'type': 'video'},
                            {'type': 'text', 'text': '\n'}]
        else:
            videos, metas = [frames], [meta(actual)]
            content = [{'type': 'video'}]
        content.append({'type': 'text', 'text': text})
        prompt = qa_processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        vp = qa_processor.video_processor
        inputs = qa_processor(text=[prompt], videos=videos, video_metadata=metas, do_sample_frames=False,
                              size=dict(shortest_edge=vp.size['shortest_edge'],
                                        longest_edge=max(len(v) for v in videos) * args.answer_max_pixels),
                              return_tensors='pt').to(qa.device)
        generated = qa.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = qa_processor.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        torch.cuda.synchronize()
        return text, dict(times=[round(t, 2) for t in actual], grid=inputs['video_grid_thw'].tolist(),
                          input_tokens=int(inputs['input_ids'].shape[1]), decode_s=decode_s,
                          answer_s=time.perf_counter() - clock)

    with shard.open('a') as stream, torch.inference_mode():
        for row in tqdm(todo, desc=f'GPU {rank}', unit='q', position=rank, dynamic_ncols=True):
            path = row['video_path']
            duration = reuse[row['id']]['duration'] if reuse else video_duration(path)

            def expand(span, depth, branch, prior):
                # one tree node; uncertain nodes split into their top peaks, confident nodes become leaves
                level = ground(path, *span, row['question'], depth)
                level.update(index=len(levels), branch=branch, prior=prior, score=prior * level['confidence'])
                levels.append(level)
                window = level['window']
                stop = None
                if level['confident']:
                    stop = 'confident'
                elif depth == args.max_depth - 1:
                    stop = 'max_depth'
                elif level['cell'] <= args.min_cell:
                    stop = 'min_cell'
                elif window[1] - window[0] >= 0.9 * (span[1] - span[0]):
                    stop = 'no_narrowing'
                elif len(levels) >= args.max_nodes:
                    stop = 'budget'
                if stop:
                    leaves.append(dict(branch=branch, last=level['index'], stop=stop))
                    return
                # prior of a child = product of the peak shares along its path
                for k, peak in enumerate([level, *level['peaks']]):
                    if len(levels) >= args.max_nodes:
                        break
                    expand(tuple(peak['window']), depth + 1, f'{branch}.{k}', prior * peak['confidence'])

            if reuse:
                old = reuse[row['id']]
                levels, leaves = old['levels'], old.get('leaves', old.get('branches'))
            else:
                levels, leaves = [], []
                expand((0.0, duration), 0, '0', 1.0)
            # rank leaves: confident first, then by score
            ranked = sorted(leaves, key=lambda b: (levels[b['last']]['confident'], levels[b['last']]['score']),
                            reverse=True)
            final = levels[ranked[0]['last']]
            stop = ranked[0]['stop']
            # widen the window around its centre, shifted back inside the video if it spills over
            lo, hi = final['window']
            width = min(duration, (hi - lo) * args.pad)
            lo = min(max(0.0, (lo + hi - width) / 2), duration - width)
            answer_window = [lo, lo + width]
            # top-K leaves (non-overlapping), optionally filled with extra coarse peaks
            windows = [answer_window]
            others = [levels[b['last']]['window'] for b in ranked[1:]]
            if args.fill_coarse:
                others += [p['window'] for p in levels[0]['peaks']]
            for w in others:
                if len(windows) >= args.top_leaves:
                    break
                if all(w[1] <= a or w[0] >= b for a, b in windows):
                    windows.append(w)
            if args.ground_only:
                prediction, system2 = None, dict(decode_s=0.0, answer_s=0.0)
            else:
                prediction, system2 = answer(path, windows, row['question'], row.get('choices'), duration)
            mcq = {}
            if args.prompt == 'mcq' and prediction is not None:
                letter = parse_letter(prediction, len(row['choices']))
                mcq = dict(choices=row['choices'], right_answer=row['right_answer'], pred_letter=letter,
                           correct=letter == row['right_answer'])
            stream.write(json.dumps(dict(
                id=row['id'], video_path=path, question=row['question'], answer=row['answer'], spans=row.get('spans'),
                prediction=prediction, **mcq, sub_category=row.get('sub_category'),
                duration=duration, confident=final['confident'], stop=stop,
                final_depth=final['depth'], final_branch=final['branch'], final_window=final['window'], answer_window=answer_window, answer_windows=windows,
                num_levels=len(levels), num_leaves=len(leaves),
                grounding_frames=sum(l['frames'] for l in levels), answer_frames=args.answer_frames,
                levels=levels, leaves=leaves, system2=system2, settings=vars(args)), ensure_ascii=False) + '\n')
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
        records = [merged[r['id']] for r in rows if r['id'] in merged]
        (out / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        write_inspect(records, out)
        print(json.dumps(summarize(records, out)))
    if world > 1:
        dist.destroy_process_group()


def summarize(records, out):
    from collections import Counter
    n = len(records) or 1
    summary = dict(
        questions=len(records),
        confident=sum(r['confident'] for r in records),
        stop_reason=dict(Counter(r['stop'] for r in records)),
        levels_used=dict(sorted(Counter(r['num_levels'] for r in records).items())),
        branched=sum(r.get('num_leaves', r.get('num_branches', 1)) > 1 for r in records),
        mean_grounding_frames=round(sum(r['grounding_frames'] for r in records) / n, 1),
        mean_final_window_s=round(sum(r['final_window'][1] - r['final_window'][0] for r in records) / n, 1),
        mean_seconds=dict(
            decode=round(sum(sum(l['decode_s'] for l in r['levels']) + r['system2']['decode_s'] for r in records) / n, 2),
            ground=round(sum(sum(l['ground_s'] for l in r['levels']) for r in records) / n, 2),
            answer=round(sum(r['system2']['answer_s'] for r in records) / n, 2)),
        predictions=str((out / 'predictions.jsonl').resolve()), inspect=str((out / 'inspect.md').resolve()))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def write_inspect(records, out):
    md = [f'# Hierarchical sampling: {len(records)} questions\n',
          'Each level: probability sparkline over its cells (▁ low … █ high), `^` = window '
          '(Kadane segment ± margin), `~` = extra peaks followed as separate branches. Windows are VideoJev predictions, not ground truth.\n']
    for k, r in enumerate(records, 1):
        md += [f'\n---\n\n## {k}. `{r["id"]}`: `{Path(r["video_path"]).name}` ({mmss(r["duration"])})\n',
               f'**Q:** {r["question"]}  ', f'**GT:** {r["answer"]}  ', f'**Prediction:** {r["prediction"]}  ',
               f'**Final window:** {mmss(r["final_window"][0])}–{mmss(r["final_window"][1])} '
               f'({r["final_window"][1] - r["final_window"][0]:.0f} s, level {r["final_depth"]}, '
               f'branch {r.get("final_branch", "0")}), '
               f'confident = {r["confident"]}, stop = {r["stop"]}\n',
               '| branch | level | span | frames | cell | segment | confidence | peak p |',
               '|---|---|---|---|---|---|---|---|']
        for l in r['levels']:
            md.append(f'| {l.get("branch", "0")} | {l["depth"]} | {mmss(l["span"][0])}–{mmss(l["span"][1])} | {l["frames"]} | {l["cell"]:.1f} s | '
                      f'{mmss(l["segment"][0])}–{mmss(l["segment"][1])} | {l["confidence"]:.2f} | {l["peak"]:.2f} |')
        md.append('\n```')
        for l in r['levels']:
            bars = ''.join(BARS[min(7, int(p * 8))] for p in l['probabilities'])
            marks = [' '] * len(l['probabilities'])
            for p in l.get('peaks', []):
                for i in range(p['window_cells'][0], p['window_cells'][1] + 1):
                    marks[i] = '~'
            for i in range(l['window_cells'][0], l['window_cells'][1] + 1):
                marks[i] = '^'
            label = f'{l.get("branch", "0")} L{l["depth"]}'
            md += [f'{label} {bars}', f'{" " * len(label)} {"".join(marks).rstrip()}']
        md.append('```')
    (out / 'inspect.md').write_text('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
