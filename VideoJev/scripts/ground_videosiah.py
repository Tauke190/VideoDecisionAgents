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
    parser.add_argument('--limit', type=int, help='Only the first N questions (for a quick test)')
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

    with shard.open('a') as stream, torch.inference_mode():
        for path in tqdm(sorted(todo), desc=f'GPU {rank}', unit='video', position=rank, dynamic_ncols=True):
            questions = [dict(id=r['id'], task='grounding', query=r['question']) for r in todo[path]]
            try:
                results = model.predict_video(processor, path, questions, fps=args.fps,
                                              max_frames=args.max_frames, context=args.context,
                                              threshold=args.threshold)
            except (av.error.FFmpegError, ValueError, RuntimeError) as error:
                with (out / f'skipped_rank{rank}.jsonl').open('a') as log:
                    log.write(json.dumps(dict(video_path=path, questions=[r['id'] for r in todo[path]],
                                              error=str(error))) + '\n')
                tqdm.write(f'GPU {rank}: skipped video {path}: {error}')
                continue
            by_id = {r['id']: r for r in results}
            for r in todo[path]:
                res = by_id[r['id']]
                probs = res.get('probabilities') or []
                stream.write(json.dumps(dict(id=r['id'], video_path=r['video_path'],
                                             interval=res['interval'],
                                             peak=round(max(probs), 4) if probs else None),
                                        ensure_ascii=False) + '\n')
                stream.flush()

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
