"""Open-ended video QA predictions from ANY Qwen-family video LMM (model-agnostic).

Same job and output format as eval_qwen25vl_open.py, but the model is a pure swap:
it loads via AutoProcessor + AutoModelForImageTextToText, so Qwen2.5-VL, Qwen3-VL and
Qwen3.5-Moe (including quantized checkpoints) all work with --model alone.

Evaluation protocol is unchanged from the matched LongVT run: uniform frame sampling
(--num-frames) and a per-frame pixel cap (--max-pixels). Thinking models emit
<think>...</think> before the answer, so give them a larger --max-new-tokens; the
judge's answer extraction keys on the <answer> tag, so --think also sends the system
prompt that asks for <think>/<answer> formatting.

Writes predictions only (no scoring); grade them with the LongVT judge afterwards.
Run under torchrun for multi-GPU sharding; each GPU answers whole videos, resumes by ID.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import numpy as np

# Reuse the exact frame samplers from the Qwen2.5-VL script (same scripts/ dir on sys.path).
from eval_qwen25vl_open import sample_frames, sample_frames_window

# LongVT's reasoning system prompt: asks the model to put its answer in <answer></answer>,
# which is what the judge's extract_answer_tag looks for. Used for --think models.
THINK_SYSTEM = (
    "You are a helpful assistant. When the user asks a question, your response must include "
    "two parts: first, the reasoning process enclosed in <think>...</think> tags, then the "
    "final answer enclosed in <answer>...</answer> tags. Please provide a clear, concise "
    "response within <answer> </answer> tags that directly addresses the question."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--out', required=True, help='Output directory for predictions.jsonl')
    parser.add_argument('--model', required=True, help='Any Qwen video LMM, e.g. Qwen/Qwen3-VL-4B-Thinking')
    parser.add_argument('--num-frames', type=int, default=64)
    parser.add_argument('--max-pixels', type=int, default=50176,
                        help='Per-frame pixel cap (LongVT eval uses 50176 = 224x224)')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--instruction', default='Answer the question directly with a short answer or phrase')
    parser.add_argument('--think', action='store_true',
                        help='Thinking model: send the <think>/<answer> system prompt and keep the full trace')
    parser.add_argument('--grounding', help='grounding.jsonl from ground_videosiah.py: sample the '
                        'num-frames budget only inside each question\'s top-1 clip interval, '
                        'instead of uniformly over the whole video')
    parser.add_argument('--limit', type=int, help='Only the first N questions (for a quick test)')
    args = parser.parse_args()
    if args.num_frames < 2 or args.num_frames % 2:
        parser.error('--num-frames must be an even number (Qwen video models pair frames)')

    import av
    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
    from transformers.video_utils import VideoMetadata

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
    shard = out / f'predictions_rank{rank}.jsonl'
    done = {json.loads(line)['id'] for line in shard.read_text().splitlines()} if shard.exists() else set()
    todo = {}
    for row in rows:
        if row['video_path'] in owned and row['id'] not in done:
            todo.setdefault(row['video_path'], []).append(row)

    # Grounded mode: {id: [t0, t1]} from stage 1. Each question samples inside its own clip,
    # so frames are drawn per question rather than once per video.
    grounding = {}
    if args.grounding:
        for line in Path(args.grounding).read_text().splitlines():
            record = json.loads(line)
            grounding[record['id']] = record['interval']
        missing = [r['id'] for rs in todo.values() for r in rs if r['id'] not in grounding]
        if missing:
            raise SystemExit(f'--grounding is missing {len(missing)} ids, e.g. {missing[:3]}; '
                             'run ground_videosiah.py over the same questions first')

    import math
    from PIL import Image
    config = AutoConfig.from_pretrained(args.model)
    model_type = config.model_type
    processor = AutoProcessor.from_pretrained(args.model)
    # Patch granularity (patch_size * spatial_merge_size) that frame dims must be a multiple of.
    vc = getattr(config, 'vision_config', None)
    factor = (getattr(vc, 'patch_size', 16) or 16) * (getattr(vc, 'spatial_merge_size', 2) or 2) if vc else 32
    load_kwargs = dict(attn_implementation='flash_attention_2', device_map={'': local})
    # Quantized checkpoints (GPTQ/FP8) carry their own compute dtype; only force bf16 otherwise.
    if getattr(config, 'quantization_config', None) is None:
        load_kwargs['dtype'] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs).eval()

    # Qwen2.5-VL honors a per-call `size` (short/long edge) to cap frame pixels. Qwen3-VL/3.5 ignore
    # it for video, so for those we resize each frame to the max_pixels budget ourselves (aspect
    # preserved, dims a multiple of `factor`) and let the processor keep them as-is.
    qwen25_size = dict(shortest_edge=min(128 * 28 * 28, args.max_pixels), longest_edge=args.max_pixels)

    def budget_frames(frames):
        if model_type == 'qwen2_5_vl':
            return frames
        h, w = frames.shape[1:3]
        scale = math.sqrt(args.max_pixels / (h * w))
        nh = max(factor, round(h * scale / factor) * factor)
        nw = max(factor, round(w * scale / factor) * factor)
        if (nh, nw) == (h, w):
            return frames
        return np.stack([np.asarray(Image.fromarray(f).resize((nw, nh), Image.BICUBIC)) for f in frames])

    def prepare(sampled):
        frames, indices, total, fps, duration = sampled
        metadata = VideoMetadata(total_num_frames=total, fps=fps,
                                 frames_indices=indices.tolist(), duration=duration)
        return metadata, budget_frames(frames), indices, fps

    def extract_answer(raw):
        # Thinking models auto-open <think> in the prompt and emit the answer AFTER </think>.
        # Keep only that post-think answer so the judge grades the answer, not the reasoning. No
        # </think> means the think block never closed within the token budget -> no answer (judge
        # scores it 0, which is fair). Residual chat specials (<|im_end|> etc.) are stripped.
        import re
        answer = raw.rsplit('</think>', 1)[1] if '</think>' in raw else ''
        return re.sub(r'<\|[^>]*\|>', '', answer).strip()
    proc_video_kwargs = dict(size=qwen25_size) if model_type == 'qwen2_5_vl' else {}

    def skip(video_path, ids, error):
        with (out / f'skipped_rank{rank}.jsonl').open('a') as log:
            log.write(json.dumps(dict(video_path=video_path, questions=ids, error=str(error))) + '\n')

    with shard.open('a') as stream, torch.inference_mode():
        for path in tqdm(sorted(todo), desc=f'GPU {rank}', unit='video', position=rank, dynamic_ncols=True):
            shared = None
            if not grounding:
                # Whole-video uniform sampling: decode once and reuse for every question.
                try:
                    shared = prepare(sample_frames(path, args.num_frames))
                except (av.error.FFmpegError, ValueError) as error:
                    skip(path, [r['id'] for r in todo[path]], error)
                    tqdm.write(f'GPU {rank}: skipped corrupt video {path}')
                    continue
            for row in todo[path]:
                if grounding:
                    # Sample the frame budget only inside this question's top-1 clip.
                    try:
                        prepared = prepare(sample_frames_window(path, args.num_frames, *grounding[row['id']]))
                    except (av.error.FFmpegError, ValueError) as error:
                        skip(path, [row['id']], error)
                        tqdm.write(f'GPU {rank}: skipped {row["id"]} in {path}')
                        continue
                else:
                    prepared = shared
                metadata, vframes, indices, fps = prepared
                content = [{'type': 'video'}, {'type': 'text', 'text': f'{row["question"]}\n{args.instruction}'}]
                messages = [{'role': 'user', 'content': content}]
                if args.think:
                    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': THINK_SYSTEM}]}] + messages
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], videos=[vframes], video_metadata=[metadata],
                                   do_sample_frames=False, return_tensors='pt', **proc_video_kwargs).to(model.device)
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                new_tokens = output[0, inputs['input_ids'].shape[1]:]
                record = dict(id=row['id'], video_path=row['video_path'], question=row['question'],
                              answer=row['answer'], model=args.model, num_frames=args.num_frames,
                              max_pixels=args.max_pixels,
                              frame_times=[round(float(i) / fps, 2) for i in indices],
                              input_tokens=int(inputs['input_ids'].shape[1]),
                              output_tokens=int(new_tokens.shape[0]))
                if grounding:
                    record['interval'] = [round(float(grounding[row['id']][0]), 2),
                                          round(float(grounding[row['id']][1]), 2)]
                if args.think:
                    # Keep special tokens so </think> is visible, then split off the answer.
                    raw = processor.decode(new_tokens, skip_special_tokens=False)
                    record['prediction'] = extract_answer(raw)
                    record['think_trace'] = raw[:4000]
                    record['answered'] = bool(record['prediction'])  # False if it never left <think>
                else:
                    record['prediction'] = processor.decode(new_tokens, skip_special_tokens=True).strip()
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {}
        for worker in range(world):
            path = out / f'predictions_rank{worker}.jsonl'
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                record = json.loads(line)
                merged[record['id']] = record
        order = [r['id'] for r in rows if r['id'] in merged]
        (out / 'predictions.jsonl').write_text(
            ''.join(json.dumps(merged[i], ensure_ascii=False) + '\n' for i in order))
        skipped = sorted({json.loads(line)['video_path'] for log in out.glob('skipped_rank*.jsonl')
                          for line in log.read_text().splitlines()})
        print(json.dumps(dict(predictions=len(order), missing=len(rows) - len(order),
                              skipped_videos=skipped, out=str(out / 'predictions.jsonl'))))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
