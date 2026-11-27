"""Stage 3 of the draft-then-verify VideoSIAH pipeline: candidate recall.

Grades EVERY drafted candidate with LongVT's exact strict scoring (score_row from
judge_videosiah_longvt.py: relax_exact_match -> math_verify -> strict 1/0 LLM judge
fallback), then reports

    recall@k = fraction of questions where at least one of the first k candidates is correct

with candidates in drafting order (greedy first, then samples), so recall@1 is the greedy
drafter's own accuracy and recall@N is the ceiling a perfect verifier (VideoJev choice
scoring) could reach. Identical candidate strings within a question are judged once.

Questions with no candidates (grounding skipped) count as misses against --questions when
given, so the headline numbers are over the full benchmark, not just the drafted subset.

Run under torchrun for multi-GPU sharding; resumes by id; rank 0 merges and summarizes.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from judge_videosiah_longvt import build_judge_prompt, parse_binary, score_row  # noqa: E402

# Caption candidates describe the clip instead of answering, so the strict answer-vs-answer
# judge does not apply; ask instead whether the caption carries the answer's information.
CONTAINS_PROMPT = """You are a strict evaluator. Decide whether a video caption contains the information needed to answer a question correctly.

Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Caption:
```
{caption}
```

Output 1 if the caption explicitly states the key information of the ground truth answer (the same object, attribute, number, text or event; wording may differ). Output 0 if the information is missing, only partially present, or contradicted. Output only 1 or 0."""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--candidates', required=True, help='candidates.jsonl from draft_videosiah.py')
    parser.add_argument('--out', required=True)
    parser.add_argument('--questions', help='Full benchmark questions.json; undrafted questions count as misses')
    parser.add_argument('--model', default='Qwen/Qwen3-8B', help='Local judge model (same as the LongVT judge run)')
    parser.add_argument('--no-judge', dest='use_judge', action='store_false', help='String matchers only')
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--criterion', choices=['auto', 'answer', 'contains'], default='auto',
                        help='answer: strict LongVT answer judge; contains: does the caption carry the GT '
                             'information; auto: contains for caption-mode drafts, answer otherwise')
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from tqdm import tqdm

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        dist.init_process_group('gloo', timeout=timedelta(hours=12))

    rows = [json.loads(line) for line in Path(args.candidates).read_text().splitlines() if line.strip()]
    if args.criterion == 'auto':
        args.criterion = 'contains' if rows and rows[0].get('mode', 'answer').startswith('caption') else 'answer'
    if args.criterion == 'contains' and not args.use_judge:
        parser.error('--criterion contains needs the LLM judge (no string matcher applies to captions)')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f'recall_rank{rank}.jsonl'
    done = {json.loads(line)['id'] for line in shard.read_text().splitlines()} if shard.exists() else set()
    todo = [r for r in rows[rank::world] if r['id'] not in done]

    if args.use_judge and todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.cuda.set_device(local)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     attn_implementation='flash_attention_2',
                                                     device_map={'': local}).eval()

        @torch.inference_mode()
        def grade(predict_str, ground_truth, question):
            # Identical to judge_videosiah_longvt.py's local backend.
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': build_judge_prompt(
                predict_str, ground_truth, question)}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            inputs = tokenizer([text], return_tensors='pt').to(model.device)
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return parse_binary(reply), reply

        @torch.inference_mode()
        def contains(caption, ground_truth, question):
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': CONTAINS_PROMPT.format(
                question=question, answer=ground_truth, caption=caption)}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer([text], return_tensors='pt').to(model.device)
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return parse_binary(reply), reply
    else:
        def grade(predict_str, ground_truth, question):
            return 0, ''

    with shard.open('a') as stream:
        for row in tqdm(todo, desc=f'rank {rank}', unit='q', position=rank, dynamic_ncols=True):
            verdicts = {}
            for candidate in row['candidates']:
                key = candidate.strip()
                if key not in verdicts and args.criterion == 'contains':
                    verdict, _ = contains(key, row['answer'], row['question'])
                    verdicts[key] = (verdict, 'llm_contains')
                elif key not in verdicts:
                    result = score_row(dict(prediction=candidate, answer=row['answer'],
                                            question=row['question']), grade, args.use_judge)
                    verdicts[key] = (int(result['acc_score'] >= 1.0), result['stage'])
            correct = [verdicts[c.strip()][0] for c in row['candidates']]
            stream.write(json.dumps(dict(id=row['id'], question=row['question'], answer=row['answer'],
                                         candidates=row['candidates'], correct=correct,
                                         stages=[verdicts[c.strip()][1] for c in row['candidates']],
                                         unique=len(verdicts)), ensure_ascii=False) + '\n')
            stream.flush()

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {}
        for path in sorted(out.glob('recall_rank*.jsonl')):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                merged[record['id']] = record
        order = [r['id'] for r in rows if r['id'] in merged]
        graded = [merged[i] for i in order]
        (out / 'recall.jsonl').write_text(''.join(json.dumps(g, ensure_ascii=False) + '\n' for g in graded))
        total = len(json.loads(Path(args.questions).read_text())) if args.questions else len(graded)
        n_max = max((len(g['correct']) for g in graded), default=0)
        ks = sorted({k for k in (1, 2, 3, 5, 9, 17, 33) if k <= n_max} | {n_max})

        def recall(k, denominator):
            return round(sum(any(g['correct'][:k]) for g in graded) / denominator, 4) if denominator else None

        summary = dict(
            questions=total, drafted=len(graded), candidates_per_question=n_max,
            recall_at_k={k: recall(k, total) for k in ks},                      # over the full benchmark
            recall_at_k_drafted={k: recall(k, len(graded)) for k in ks},        # over drafted questions only
            greedy_accuracy=recall(1, total),
            mean_unique_candidates=round(sum(g['unique'] for g in graded) / len(graded), 2) if graded else None,
            mean_correct_fraction=round(sum(sum(g['correct']) / len(g['correct']) for g in graded)
                                        / len(graded), 4) if graded else None,
            judge=args.model if args.use_judge else None, criterion=args.criterion,
            source=str(args.candidates))
        if graded and all('sources' in merged_row for merged_row in rows if merged_row['id'] in merged):
            # Per Kadane rank: how often that interval's own candidate(s) are correct, and how often
            # it is the FIRST interval that contains a correct candidate.
            src = {r['id']: r['sources'] for r in rows}
            per_rank, first_hit = {}, {}
            for g in graded:
                ranks = [s['clip_rank'] for s in src[g['id']]]
                hits = {}
                for r_, ok in zip(ranks, g['correct']):
                    hits[r_] = hits.get(r_, 0) or ok
                for r_, ok in hits.items():
                    per_rank.setdefault(r_, []).append(ok)
                hit_ranks = sorted(r_ for r_, ok in hits.items() if ok)
                key = hit_ranks[0] if hit_ranks else 'none'
                first_hit[key] = first_hit.get(key, 0) + 1
            summary['interval_accuracy_by_rank'] = {r_: round(sum(v) / len(v), 4) for r_, v in sorted(per_rank.items())}
            summary['first_correct_interval_rank'] = {str(k): v for k, v in sorted(first_hit.items(), key=lambda kv: str(kv[0]))}
        (out / 'recall_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
