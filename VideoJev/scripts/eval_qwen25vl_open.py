"""Open-ended video QA predictions from Qwen2.5-VL with sparse uniform frame sampling.

Writes predictions only (no scoring); grade them later with an LLM judge.
Input: a JSON list of {"id", "video_path", "question", "answer"} (video_path relative to the file).
Run with torchrun for multiple GPUs; each GPU answers whole videos and the run resumes by ID.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import numpy as np


def fitted(frame, width, height):
    """RGB frame at the stream's size. Merged videos (e.g. cooking_103_merged.mp4: 1280x720 clips then 1280x960)
    change resolution mid-stream; such frames are scaled to fit, keeping aspect ratio, and padded with black."""
    if (frame.width, frame.height) == (width, height):
        return frame.to_ndarray(format='rgb24')
    scale = min(width/frame.width, height/frame.height)
    w, h = max(2, round(frame.width*scale)), max(2, round(frame.height*scale))
    canvas = np.zeros((height, width, 3), np.uint8)
    top, left = (height-h)//2, (width-w)//2
    canvas[top:top+h, left:left+w] = frame.to_ndarray(format='rgb24', width=w, height=h)
    return canvas


def sample_frames(path, num_frames):
    """Seek to num_frames uniformly spaced frame indices over the whole video (RGB uint8)."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type, stream.thread_count = 'AUTO', 4
        fps = float(stream.average_rate or stream.guessed_rate)
        duration = (float(stream.duration*stream.time_base) if stream.duration else
                    float(container.duration/av.time_base))
        total = stream.frames or int(duration*fps)
        indices = np.linspace(0, total-1, num_frames).round().astype(int)
        start = float(stream.start_time*stream.time_base) if stream.start_time is not None else 0.
        # The decoder's size follows the current frame, so fix the output size from the header first.
        width, height = stream.codec_context.width, stream.codec_context.height
        frames, last = [], None
        for index in indices:
            target = start + index/fps
            container.seek(int(target/stream.time_base), stream=stream, backward=True, any_frame=False)
            frame = None
            for candidate in container.decode(stream):
                frame = candidate
                if candidate.pts is not None and float(candidate.pts*stream.time_base) >= target - .5/fps:
                    break
            # A seek past the final keyframe can yield nothing; reuse the previous frame.
            last = fitted(frame, width, height) if frame is not None else last
            if last is None:
                raise ValueError(f'Could not decode frames: {path}')
            frames.append(last)
    return np.stack(frames), indices, total, fps, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--out', required=True, help='Output directory for predictions.jsonl')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--num-frames', type=int, default=64)
    parser.add_argument('--max-pixels', type=int, default=768*28*28,
                        help="Pixel cap per frame (default: Qwen2.5-VL's video default; peaks ~21.5 GB at 64 frames)")
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--instruction', default='Answer the question based on the video.')
    parser.add_argument('--limit', type=int, help='Only the first N questions (for a quick test)')
    args = parser.parse_args()
    if args.num_frames < 2 or args.num_frames % 2:
        parser.error('--num-frames must be an even number (Qwen2.5-VL pairs frames)')

    import av
    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.video_utils import VideoMetadata

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        # Ranks finish at different times; the barrier only waits for the merge.
        dist.init_process_group('gloo', timeout=timedelta(hours=12))

    source = Path(args.questions)
    rows = json.loads(source.read_text())[:args.limit]
    for row in rows:
        row['video_path'] = str((source.parent/row['video_path']).resolve())
    videos = sorted({r['video_path'] for r in rows})
    owned = set(videos[rank::world])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out/f'predictions_rank{rank}.jsonl'
    done = {json.loads(line)['id'] for line in shard.read_text().splitlines()} if shard.exists() else set()
    todo = {}
    for row in rows:
        if row['video_path'] in owned and row['id'] not in done:
            todo.setdefault(row['video_path'], []).append(row)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map={'': local}).eval()
    with shard.open('a') as stream, torch.inference_mode():
        for path in tqdm(sorted(todo), desc=f'GPU {rank}', unit='video', position=rank, dynamic_ncols=True):
            try:
                frames, indices, total, fps, duration = sample_frames(path, args.num_frames)
            except (av.error.FFmpegError, ValueError) as error:
                # Undecodable source file (e.g. corrupt 6lRTHAIJDIU.mp4): skip it and record why.
                with (out/f'skipped_rank{rank}.jsonl').open('a') as log:
                    log.write(json.dumps(dict(video_path=path, questions=[r['id'] for r in todo[path]],
                                              error=str(error)))+'\n')
                tqdm.write(f'GPU {rank}: skipped corrupt video {path}')
                continue
            metadata = VideoMetadata(total_num_frames=total, fps=fps, frames_indices=indices.tolist(), duration=duration)
            for row in todo[path]:
                messages = [{'role': 'user', 'content': [
                    {'type': 'video'}, {'type': 'text', 'text': f'{row["question"]}\n{args.instruction}'}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # The video processor ignores max_pixels; the per-frame cap must be passed as size.
                size = dict(shortest_edge=min(128*28*28, args.max_pixels), longest_edge=args.max_pixels)
                inputs = processor(text=[prompt], videos=[frames], video_metadata=[metadata], do_sample_frames=False,
                                   size=size, return_tensors='pt').to(model.device)
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                prediction = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                stream.write(json.dumps(dict(id=row['id'], video_path=row['video_path'], question=row['question'],
                    answer=row['answer'], prediction=prediction.strip(), model=args.model,
                    num_frames=args.num_frames, frame_times=[round(float(i)/fps, 2) for i in indices],
                    input_tokens=int(inputs['input_ids'].shape[1])), ensure_ascii=False)+'\n')
                stream.flush()

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {}
        for worker in range(world):
            for line in (out/f'predictions_rank{worker}.jsonl').read_text().splitlines():
                record = json.loads(line)
                merged[record['id']] = record
        order = [r['id'] for r in rows if r['id'] in merged]
        (out/'predictions.jsonl').write_text(''.join(json.dumps(merged[i], ensure_ascii=False)+'\n' for i in order))
        skipped = sorted({json.loads(line)['video_path'] for log in out.glob('skipped_rank*.jsonl')
                          for line in log.read_text().splitlines()})
        print(json.dumps(dict(predictions=len(order), missing=len(rows) - len(order),
                              skipped_videos=skipped, out=str(out/'predictions.jsonl'))))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
