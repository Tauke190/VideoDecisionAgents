"""Stage 1 of the grounded-clip VideoSIAH pipeline: VideoJev-4B temporal grounding.

For each (video, question) VideoSIAH item, run VideoJev grounding with the RAW question
text as the grounding query and record the top-1 clip -- the single max-likelihood
interval the grounder returns (videojev_temporal.decode is Kadane over per-window logits,
so there is exactly one interval per query; that interval IS the top-1 clip).

Output grounding.jsonl ({"id", "video_path", "interval":[t0,t1], "peak"}) feeds stage 2
(scripts/eval_video_open.py --grounding), which samples the answering model's frames only
inside [t0, t1].

Grounding regime follows docs/EVALUATION.md: fps=4, context=147456, threshold=0.25,
max_frames=2048. VideoSIAH videos run 15-80 min, so grounding falls back to uniform
full-video sampling (videojev_video.sampling_grid) and the returned interval is still in
real seconds over the full duration.

Run under torchrun for multi-GPU sharding; each GPU grounds whole videos and the run
resumes by id. One video is encoded once and all its questions share that encoding.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--out', required=True, help='Output directory for grounding.jsonl')
    parser.add_argument('--checkpoint', default='ZhuYuhan/VideoJev-4B',
                        help='VideoJev HF checkpoint (repo id or local dir)')
    parser.add_argument('--fps', type=float, default=4.0, help='Grounding fps (official: 4)')
    parser.add_argument('--max-frames', type=int, default=2048, help='Frame cap (official: 2048)')
    parser.add_argument('--context', type=int, default=147456, help='Grounding context tokens (official: 147456)')
    parser.add_argument('--threshold', type=float, default=0.25, help='Grounding threshold (official: 0.25)')
    parser.add_argument('--min-context', type=int, default=32768,
                        help='On CUDA OOM the context shrinks by 2/3 per retry, down to this floor')
    parser.add_argument('--video-features', help='Directory to cache each video\'s WHOLE-video native features '
                        '(<video stem>.pt, ~377 MB for 2048 frames at 98k context) for later VideoJev passes')
    parser.add_argument('--clip-features', help='Directory to cache each question\'s top-K clips\' native '
                        'vision features (<id>.pt) for draft_videosiah.py')
    parser.add_argument('--decode-max-edge', type=int, default=1280,
                        help='Downscale frames at decode time to this long edge (0 = official full-res decode; '
                             '4K videos then need ~100 GB RAM per process)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Disjoint intervals per query via iterated Kadane (#1 = official top-1)')
    parser.add_argument('--interval-gap', type=float, default=5.0,
                        help='Seconds masked on each side of a found interval so the next one is a distinct moment')
    parser.add_argument('--min-clip-groups', type=int, default=8,
                        help='Widen short clips to at least this many temporal groups (2 frames each)')
    parser.add_argument('--max-clip-groups', type=int, default=32,
                        help='Uniformly subsample long clips to at most this many temporal groups')
    parser.add_argument('--max-pixels', type=int, default=640 * 640,
                        help='Per-frame pixel ceiling for the whole-video pass (official 640*640; 50176 = 224*224 '
                             'matches LongVT). The context budget can still lower it further')
    parser.add_argument('--clip-max-pixels', type=int,
                        help='If set, each clip\'s features for the drafter are RE-ENCODED from the same decoded '
                             'frames at this per-frame ceiling instead of sliced from the whole-video features '
                             '(no extra frames, higher resolution)')
    parser.add_argument('--limit', type=int, help='Only the first N questions (for a quick test)')
    args = parser.parse_args()

    import av
    import numpy as np
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
    rows = json.loads(source.read_text())
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        row['video_path'] = str((source.parent / row['video_path']).resolve())
    videos = sorted({r['video_path'] for r in rows})
    owned = set(videos[rank::world])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f'grounding_rank{rank}.jsonl'
    done = {json.loads(line)['id'] for line in shard.read_text().splitlines()} if shard.exists() else set()
    todo = {}
    for row in rows:
        if row['video_path'] in owned and row['id'] not in done:
            todo.setdefault(row['video_path'], []).append(row)

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
                                      device_map={'': local}, attn_implementation='flash_attention_2')
    model.prepare_for_inference()

    # predict_video == prepare_video -> encode_video -> predict_features; unrolled here so the
    # native vision features stay in hand and the top clip's slice can be cached for a drafter.
    import importlib
    temporal = importlib.import_module(type(model).__module__.rsplit('.', 1)[0] + '.videojev_temporal')
    processing = importlib.import_module(type(model).__module__.rsplit('.', 1)[0] + '.processing_videojev')
    keep = {} if args.clip_max_pixels else None  # decoded frames, for --clip-max-pixels re-encoding
    decoder = dict(model.config.decoder, threshold=args.threshold)

    if args.decode_max_edge:
        # The official reader keeps all (up to 2048) frames at FULL resolution before the processor
        # shrinks them to ~200-400 px: a 4K video is ~47 GB of uint8 per process (x2 while stacking),
        # which got the job OOM-killed. Rebuild read_video from its own source with frames
        # downscaled at decode time (long edge <= --decode-max-edge, aspect kept). Videos already
        # within the cap are decoded bit-identically to the official path.
        import inspect
        video_module = importlib.import_module(type(model).__module__.rsplit('.', 1)[0] + '.videojev_video')
        source = inspect.getsource(video_module.read_video)
        if source.count(".to_ndarray(format='rgb24')") != 2:
            raise RuntimeError('read_video changed upstream; refusing to patch frame decoding blindly')
        cap = args.decode_max_edge
        target = {}  # per-video output size, fixed by its first frame

        def capped_rgb(frame):
            # Some VideoSIAH videos are concatenations whose resolution changes mid-stream
            # (e.g. 1280x720 then 1280x960), which the official reader cannot stack; every
            # frame is therefore resized to the first frame's (capped) size.
            if 'size' not in target:
                scale = min(1., cap / max(frame.width, frame.height))
                target['size'] = (max(2, round(frame.width * scale / 2) * 2) if scale < 1 else frame.width,
                                  max(2, round(frame.height * scale / 2) * 2) if scale < 1 else frame.height)
            width, height = target['size']
            if (frame.width, frame.height) == (width, height):
                return frame.to_ndarray(format='rgb24')
            return frame.to_ndarray(format='rgb24', width=width, height=height)
        namespace = dict(vars(video_module), _capped_rgb=capped_rgb)
        exec(compile(source.replace("chosen.to_ndarray(format='rgb24')", '_capped_rgb(chosen)')
                     .replace("previous.to_ndarray(format='rgb24')", '_capped_rgb(previous)'),
                     video_module.__file__, 'exec'), namespace)
        patched_read_video = namespace['read_video']

        def read_video(*a, **k):
            target.clear()
            return patched_read_video(*a, **k)
        namespace['read_video'] = read_video  # its own recursive retry also resets the size
        processing.read_video = read_video  # prepare_video resolves read_video here
    # After the decode patch: the rebuilt prepare_video must resolve the capped read_video.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'VideoJevOpenQA'))
    from videojev_runtime import encode_frames, make_prepare_video
    prepare_video = make_prepare_video(model, args.max_pixels, keep)
    feature_dir = Path(args.clip_features) if args.clip_features else None
    if feature_dir:
        feature_dir.mkdir(parents=True, exist_ok=True)

    def top_k_intervals(logits, record):
        """Iterated Kadane: the top-1 segment, mask it (+/- --interval-gap s), re-run, K times.

        Same threshold shift, smoothing and round/repair/clamp postprocessing as
        videojev_temporal.decode, and the unmasked first pass is that exact algorithm (asserted),
        so interval #1 is the official top-1. Masked cells break runs, so later segments can
        never bridge into earlier ones. Score = Kadane sum of shifted logits (log-likelihood
        gain over background); later segments may be <= 0, i.e. weaker than background.
        """
        bounds = np.asarray(record['score_bounds'], dtype=np.float64)
        duration = record['sampled_duration']
        z = np.asarray(logits, dtype=np.float64)
        smooth = decoder['smooth']
        if smooth > 1:
            z = np.convolve(np.pad(z, (smooth // 2, smooth // 2), mode='edge'), np.ones(smooth) / smooth, mode='valid')
        z = z - np.log(decoder['threshold'] / (1 - decoder['threshold']))
        probs = 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))
        masked = np.zeros(len(z), dtype=bool)
        found = []
        while len(found) < args.top_k and not masked.all():
            best, current, start, span = -np.inf, None, 0, None
            for i in range(len(z)):
                if masked[i]:
                    current = None
                    continue
                if current is None or current <= 0:
                    start, current = i, float(z[i])
                else:
                    current += float(z[i])
                if current > best:
                    best, span = current, (start, i)
            a, b = span
            raw = [float(bounds[a, 0]), float(bounds[b, 1])]
            interval = temporal.postprocess_interval(raw, z, bounds, duration, method=decoder['postprocess'])
            found.append(dict(interval=interval, score=round(best, 4), cells=[a, b],
                              mean_prob=round(float(probs[a:b + 1].mean()), 4),
                              peak=round(float(probs[a:b + 1].max()), 4)))
            masked |= (bounds[:, 1] > raw[0] - args.interval_gap) & (bounds[:, 0] < raw[1] + args.interval_gap)
        official = temporal.decode(logits, record['score_times'], duration, decoder['threshold'], smooth,
                                   bounds=record['score_bounds'], postprocess=decoder['postprocess'])
        if found[0]['interval'] != official:
            raise RuntimeError(f'iterated Kadane #1 {found[0]["interval"]} != decode {official}')
        return found

    def ground(path, questions):
        """Ground at --context, shrinking it on CUDA OOM (24 GB cards cannot hold 147k)."""
        context = args.context
        while True:
            try:
                pixels, record = prepare_video(processor, path, fps=args.fps,
                                               max_frames=args.max_frames, context=context)
                features = model.encode_video(pixels, torch.tensor([record['grid_thw']]))
                del pixels
                # == model.predict_features, but keeping each query's raw per-cell logits.
                results = []
                for batch, rows in processing.prepare_query_groups(processor, record, questions, context):
                    output = model(**batch, visual_features=features)
                    for i, row in enumerate(rows):
                        count = int(output.score_mask[i].sum())
                        logits = output.logits[i, :count].float().cpu().tolist()
                        results.append(dict(id=row['id'], intervals=top_k_intervals(logits, record)))
                return results, features, record, context
            except torch.OutOfMemoryError:
                pixels = features = None
                torch.cuda.empty_cache()
                if context // 2 < args.min_context:
                    raise
                context = (context * 2 // 3 + 63) // 64 * 64
                tqdm.write(f'GPU {rank}: OOM on {Path(path).name}, retrying at context={context}')

    def clip_slice(qid, interval, features, record):
        """Native merger outputs of the temporal groups covering one grounded interval.

        Each VideoJev score cell is one Qwen temporal group (2 frames), and the vision tower
        encodes groups independently, so a group subset is exactly what Qwen3.5-4B (bit-identical
        vision weights) would compute on those frames. Short clips are widened symmetrically to
        --min-clip-groups; long clips are subsampled uniformly to --max-clip-groups.
        """
        bounds = record['score_bounds']
        groups = len(bounds)
        t, h, w = record['grid_thw']
        per_group = h * w // 4
        if t != groups or len(features) != groups * per_group:
            raise ValueError(f'{qid}: grid {record["grid_thw"]} disagrees with {groups} score cells')
        t0, t1 = interval
        inside = [g for g, (a, b) in enumerate(bounds) if b > t0 and a < t1]
        if not inside:  # degenerate interval: nearest cell
            mid = (t0 + t1) / 2
            inside = [min(range(groups), key=lambda g: abs(record['score_times'][g] - mid))]
        lo, hi = inside[0], inside[-1] + 1
        while hi - lo < min(args.min_clip_groups, groups):
            if lo > 0:
                lo -= 1
            if hi - lo < args.min_clip_groups and hi < groups:
                hi += 1
        chosen = list(range(lo, hi))
        if len(chosen) > args.max_clip_groups:
            chosen = sorted({chosen[int(i)] for i in np.linspace(0, len(chosen) - 1, args.max_clip_groups).round()})
        if args.clip_max_pixels:
            # Same frames as the whole-video pass (group g = frames 2g, 2g+1), re-encoded sharper.
            raw = keep['sample']['video']
            frames = raw[[min(2 * g + k, len(raw) - 1) for g in chosen for k in (0, 1)]]
            clip_features, (ct, ch, cw) = encode_frames(model, processor, frames, args.clip_max_pixels)
            if ct != len(chosen):
                raise ValueError(f'{qid}: re-encoded {ct} groups, expected {len(chosen)}')
            clip_features, h, w = clip_features.to(torch.bfloat16).cpu().contiguous(), ch, cw
        else:
            index = torch.cat([torch.arange(g * per_group, (g + 1) * per_group) for g in chosen])
            clip_features = features[index.to(features.device)].to(torch.bfloat16).cpu().contiguous()
        return dict(features=clip_features, grid_hw=[h, w], groups=chosen, times=[record['score_times'][g] for g in chosen],
                    span=[bounds[chosen[0]][0], bounds[chosen[-1]][1]], interval=list(interval))

    with shard.open('a') as stream, torch.inference_mode():
        for path in tqdm(sorted(todo), desc=f'GPU {rank}', unit='video', position=rank, dynamic_ncols=True):
            questions = [dict(id=r['id'], task='grounding', query=r['question']) for r in todo[path]]
            try:
                results, features, record, context = ground(path, questions)
            except (av.error.FFmpegError, ValueError, RuntimeError) as error:
                torch.cuda.empty_cache()
                with (out / f'skipped_rank{rank}.jsonl').open('a') as log:
                    log.write(json.dumps(dict(video_path=path, questions=[r['id'] for r in todo[path]],
                                              error=str(error))) + '\n')
                tqdm.write(f'GPU {rank}: skipped video {path}: {error}')
                continue
            by_id = {r['id']: r for r in results}
            for r in todo[path]:
                intervals = by_id[r['id']]['intervals']
                # `interval` stays the official top-1 so eval_video_open.py --grounding still works.
                entry = dict(id=r['id'], video_path=r['video_path'], interval=intervals[0]['interval'],
                             peak=intervals[0]['peak'], intervals=intervals, context=context,
                             frames=record['frames'], grid_thw=record['grid_thw'])
                if feature_dir:
                    clips = [clip_slice(r['id'], it['interval'], features, record) for it in intervals]
                    for rank_, (clip, it) in enumerate(zip(clips, intervals)):
                        clip.update(rank=rank_, score=it['score'])
                    torch.save(dict(clips=clips, fps=record['fps'], frames=record['frames'],
                                    protocol='videojev4b_native_merger_bf16_topk_clips_v2'),
                               feature_dir / f'{r["id"].replace("/", "__")}.pt')
                    entry['clip_groups'] = [len(c['groups']) for c in clips]
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                stream.flush()
            if args.video_features:
                # Whole-video native features + VideoJev's video record, so later VideoJev passes on the
                # same video (e.g. VideoJevOpenQA/select_videosiah.py) skip decode/resize/vision encode.
                # Written atomically (tmp + rename) so a killed run never leaves a truncated cache.
                video_dir = Path(args.video_features)
                video_dir.mkdir(parents=True, exist_ok=True)
                cache_file = video_dir / f'{Path(path).stem}.pt'
                tmp = cache_file.with_suffix('.pt.tmp')
                torch.save(dict(features=features.to(torch.bfloat16).cpu().contiguous(), record=record,
                                context=context, fps=args.fps, max_frames=args.max_frames,
                                max_pixels=args.max_pixels,
                                decode_max_edge=args.decode_max_edge, checkpoint=args.checkpoint,
                                video_path=path, protocol='videojev4b_native_merger_bf16_video_v1'), tmp)
                tmp.replace(cache_file)
            del features
            torch.cuda.empty_cache()

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {}
        for worker in range(world):
            path = out / f'grounding_rank{worker}.jsonl'
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                record = json.loads(line)
                merged[record['id']] = record
        order = [r['id'] for r in rows if r['id'] in merged]
        (out / 'grounding.jsonl').write_text(
            ''.join(json.dumps(merged[i], ensure_ascii=False) + '\n' for i in order))
        skipped = sorted({json.loads(line)['video_path'] for log in out.glob('skipped_rank*.jsonl')
                          for line in log.read_text().splitlines()})
        print(json.dumps(dict(grounded=len(order), missing=len(rows) - len(order),
                              skipped_videos=skipped, out=str(out / 'grounding.jsonl'))))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
