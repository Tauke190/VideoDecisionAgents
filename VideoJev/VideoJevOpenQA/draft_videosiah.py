"""Stage 2 of the draft-then-verify VideoSIAH pipeline: draft open-ended answer candidates.

By default (--mode caption) the drafter writes concise CAPTIONS of each clip, steered by the
question but not answering it; --mode answer restores short direct answers.

Qwen3.5-4B (VideoJev-4B's base; its vision tower is bit-identical to VideoJev-4B's) drafts
one candidate per grounded interval: stage 1 returns the top-K DISJOINT intervals per question
(iterated Kadane, #1 = official top-1), and candidate i is drafted from interval i alone, so
the candidates represent different moments of the video rather than resamples of one clip.
It never re-decodes or re-encodes video: `ground_videosiah.py --clip-features DIR` cached the
native merger outputs of each interval's temporal groups, injected in place of Qwen's own
vision forward. The prompt keeps Qwen's native layout, one `<t seconds>` + video-token block
per group, with the groups' real timestamps in the full video.

Candidates are written in Kadane rank order (greedy per interval, optionally followed by
--samples-per-clip sampled drafts), which candidate_recall_videosiah.py uses for recall@k.
Thinking is disabled.

Run under torchrun for multi-GPU sharding; resumes by id; rank 0 merges candidates.jsonl.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path


PROMPTS = {
    # The drafter's job is to describe the grounded clip, not to answer: VideoJev (or a later
    # step) decides. The question only steers WHICH details the caption mentions.
    'caption': ('Write one concise caption (a single sentence) describing what is visible in this clip, '
                'focusing on the people, objects, actions and any on-screen text relevant to the question '
                'below. Describe what you see; do not answer the question directly.\n'
                'Question: {question}'),
    'caption_generic': ('Write one concise caption (a single sentence) describing what is visible in this clip: '
                        'the people, objects, actions and any on-screen text.'),
    'answer': '{question}\nAnswer the question directly with a short answer or phrase',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--grounding', required=True, help='grounding.jsonl from ground_videosiah.py')
    parser.add_argument('--clip-features', required=True, help='--clip-features dir from ground_videosiah.py')
    parser.add_argument('--out', required=True, help='Output directory for candidates.jsonl')
    parser.add_argument('--model', default='Qwen/Qwen3.5-4B')
    parser.add_argument('--max-clips', type=int, help='Use only the top-N grounded intervals (default: all saved)')
    parser.add_argument('--samples-per-clip', type=int, default=0,
                        help='Extra sampled drafts per interval after its greedy draft (default 0: one candidate per interval)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--mode', choices=list(PROMPTS), default='caption',
                        help='caption: one concise caption of the clip, guided by the question but not answering '
                             'it; caption_generic: concise caption without the question; answer: short direct answer')
    parser.add_argument('--max-new-tokens', type=int, help='Default: 96 for caption modes, 64 for answer')
    parser.add_argument('--instruction', help='Override the --mode prompt; {question} is filled in')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, help='Only the first N questions (for a quick test)')
    args = parser.parse_args()
    prompt_template = args.instruction or PROMPTS[args.mode]
    if args.max_new_tokens is None:
        args.max_new_tokens = 64 if args.mode == 'answer' else 96

    import numpy as np
    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.video_utils import VideoMetadata

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('gloo', timeout=timedelta(hours=12))

    rows = json.loads(Path(args.questions).read_text())
    if args.limit:
        rows = rows[:args.limit]
    grounding = {}
    for line in Path(args.grounding).read_text().splitlines():
        record = json.loads(line)
        grounding[record['id']] = record
    feature_dir = Path(args.clip_features)
    feature_path = lambda qid: feature_dir / f'{qid.replace("/", "__")}.pt'
    usable = [r for r in rows if r['id'] in grounding and feature_path(r['id']).exists()]
    if len(usable) < len(rows):
        print(f'rank {rank}: {len(rows) - len(usable)} questions have no grounding/clip features; skipping them')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f'candidates_rank{rank}.jsonl'
    done = {json.loads(line)['id'] for line in shard.read_text().splitlines()} if shard.exists() else set()
    todo = [r for r in usable[rank::world] if r['id'] not in done]

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': local}).eval()
    vp = processor.video_processor
    patch, merge, temporal = vp.patch_size, vp.merge_size, vp.temporal_patch_size

    # Replace Qwen's vision forward with the cached VideoJev features. The processor still
    # builds the exact placeholder tokens / timestamps from a pixel-free stand-in clip of the
    # same grid; generate()'s num_return_sequences expansion repeats grid rows, so return one
    # copy of the clip's features per grid row.
    cached = {}

    def cached_video_features(pixel_values_videos, video_grid_thw=None, **kwargs):
        feats, grid = cached['features'], cached['grid']
        for row in video_grid_thw.tolist():
            if row != grid:
                raise ValueError(f'video grid {row} != cached clip grid {grid}')
        return BaseModelOutputWithPooling(pooler_output=tuple(feats for _ in range(len(video_grid_thw))))
    model.model.get_video_features = cached_video_features

    def build_inputs(row, clip):
        h, w = clip['grid_hw']
        groups = len(clip['groups'])
        # Stand-in frames only fix the grid; their pixels never reach the model.
        frames = np.zeros((groups * temporal, h * patch, w * patch, 3), dtype=np.uint8)
        # fps=1 with frame "indices" = seconds makes the processor print each group's real time.
        stamps = [t for t in clip['times'] for _ in range(temporal)]
        metadata = VideoMetadata(total_num_frames=len(stamps), fps=1.0, frames_indices=stamps,
                                 duration=float(clip['span'][1]))
        content = [{'type': 'video'}, {'type': 'text', 'text': prompt_template.format(question=row['question'])}]
        prompt = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[prompt], videos=[frames], video_metadata=[metadata], do_sample_frames=False,
                           do_resize=False, return_tensors='pt')
        grid = inputs['video_grid_thw'][0].tolist()
        if grid != [groups, h, w]:
            raise ValueError(f'{row["id"]}: processor grid {grid} != cached [{groups}, {h}, {w}]')
        n_tokens = int((inputs['input_ids'] == model.config.video_token_id).sum())
        if n_tokens != len(clip['features']):
            raise ValueError(f'{row["id"]}: {n_tokens} video tokens vs {len(clip["features"])} cached features')
        return inputs.to(model.device), grid

    def decode(sequences, prompt_len):
        return [processor.decode(s[prompt_len:], skip_special_tokens=True).strip() for s in sequences]

    torch.manual_seed(args.seed + rank)
    with shard.open('a') as stream, torch.inference_mode():
        for row in tqdm(todo, desc=f'GPU {rank}', unit='q', position=rank, dynamic_ncols=True):
            saved = torch.load(feature_path(row['id']), map_location='cpu')
            clips = saved['clips'] if 'clips' in saved else [dict(saved, rank=0, score=None)]  # v1: top-1 only
            clips = clips[:args.max_clips] if args.max_clips else clips
            candidates, sources = [], []
            # One greedy draft per grounded interval, in Kadane rank order, so candidate i
            # represents interval i (optionally followed by --samples-per-clip sampled drafts).
            for clip in clips:
                inputs, grid = build_inputs(row, clip)
                cached.update(features=clip['features'].to(model.device), grid=grid)
                prompt_len = inputs['input_ids'].shape[1]
                drafts = decode(model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False),
                                prompt_len)
                # generate() replicates the prompt num_return_sequences times BEFORE prefill, so a long
                # high-res clip x 8 samples can OOM a 24 GB card; on OOM, halve the chunk and continue.
                left, chunk = args.samples_per_clip, args.samples_per_clip
                while left:
                    n = min(chunk, left)
                    try:
                        drafts += decode(model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                        do_sample=True, temperature=args.temperature,
                                                        top_p=args.top_p, top_k=0, num_return_sequences=n),
                                         prompt_len)
                        left -= n
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if n == 1:
                            raise
                        chunk = max(1, n // 2)
                        tqdm.write(f'GPU {rank}: OOM on {row["id"]} ({prompt_len} tokens); sampling {chunk} at a time')
                for j, text in enumerate(drafts):
                    candidates.append(text)
                    sources.append(dict(clip_rank=clip['rank'], kind='greedy' if j == 0 else 'sample',
                                        interval=clip['interval'], score=clip['score'],
                                        clip_span=[round(clip['span'][0], 2), round(clip['span'][1], 2)],
                                        clip_groups=len(clip['groups']), input_tokens=prompt_len))
            stream.write(json.dumps(dict(
                id=row['id'], question=row['question'], answer=row['answer'], model=args.model,
                candidates=candidates, sources=sources, num_clips=len(clips), mode=args.mode,
                prompt=prompt_template,
                samples_per_clip=args.samples_per_clip, temperature=args.temperature,
                top_p=args.top_p), ensure_ascii=False) + '\n')
            stream.flush()

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {}
        for path in sorted(out.glob('candidates_rank*.jsonl')):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                merged[record['id']] = record
        order = [r['id'] for r in rows if r['id'] in merged]
        (out / 'candidates.jsonl').write_text(
            ''.join(json.dumps(merged[i], ensure_ascii=False) + '\n' for i in order))
        print(json.dumps(dict(drafted=len(order), missing=len(rows) - len(order),
                              out=str(out / 'candidates.jsonl'))))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
