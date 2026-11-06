"""Grade open-ended VideoSIAH predictions with a Qwen3 LLM judge.

Reads a predictions JSONL (one object per line with "id", "question", "answer",
"prediction"; the shape written by eval_qwen25vl_open.py) and asks a text-only
Qwen3 model whether each prediction matches the reference answer. Writes one
judgment per line and prints an accuracy summary.

The judge is deterministic (greedy, thinking disabled) so a re-run reproduces
the same verdicts. Run with torchrun for multiple GPUs; each GPU grades a shard
of the rows and the run resumes by ID, then rank 0 merges and scores.
"""
import argparse
from datetime import timedelta
import json
import os
import re
from pathlib import Path

SYSTEM = (
    'You are a strict but fair evaluator for an open-ended video question-answering '
    'benchmark. You are given a question, the reference (ground-truth) answer, and a '
    'model\'s predicted answer. Decide whether the prediction is correct.\n'
    'Rules:\n'
    '- The prediction is correct when it conveys the same essential meaning as the '
    'reference answer, even if it is worded differently, shorter, or adds extra correct detail.\n'
    '- It is incorrect when it contradicts the reference, misses the key fact the '
    'question asks for, or is vague/evasive where a specific answer is required.\n'
    '- Judge only against the reference answer; do not invent facts about the video.\n'
    'Reply with a single JSON object and nothing else, in the form:\n'
    '{"correct": true or false, "score": an integer 0-5, "reason": "one short sentence"}\n'
    'where score is 0 for completely wrong and 5 for a fully correct answer.'
)


def build_prompt(row):
    return (
        f'Question: {row["question"]}\n'
        f'Reference answer: {row["answer"]}\n'
        f'Model prediction: {row["prediction"]}\n\n'
        'Grade the prediction now.'
    )


def parse_verdict(text):
    """Pull {correct, score, reason} out of the judge's reply, tolerating noise."""
    body = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    correct, score, reason = None, None, ''
    for match in re.finditer(r'\{.*?\}', body, flags=re.DOTALL):
        try:
            obj = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if 'correct' in obj:
            correct = bool(obj['correct'])
            if isinstance(obj.get('score'), (int, float)):
                score = int(obj['score'])
            reason = str(obj.get('reason', ''))[:500]
    if correct is None:  # Model ignored the JSON format; sniff an explicit verdict.
        low = body.lower()
        if re.search(r'\b(incorrect|wrong)\b', low) or 'not correct' in low or 'does not match' in low:
            correct = False
        elif re.search(r'\bcorrect\b', low) or 'matches the reference' in low:
            correct = True
        # Anything else stays None: counted as a parse failure, not a guessed verdict.
    return correct, score, reason, body


def load_rows(path):
    rows, seen = [], set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record['id'] in seen:  # Later shards/merges can repeat an ID; keep the first.
            continue
        seen.add(record['id'])
        for field in ('id', 'question', 'answer', 'prediction'):
            if field not in record:
                raise KeyError(f'Prediction {record.get("id", "?")} is missing "{field}"')
        rows.append(record)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True,
                        help='Predictions JSONL (e.g. runs/.../predictions.jsonl or predictions_rank0.jsonl)')
    parser.add_argument('--out', required=True, help='Output directory for judgments.jsonl')
    parser.add_argument('--model', default='Qwen/Qwen3-8B', help='Text Qwen3 judge model')
    parser.add_argument('--think', action='store_true',
                        help='Let the judge reason in <think> blocks before deciding (slower)')
    parser.add_argument('--max-new-tokens', type=int, default=0,
                        help='Decode budget per verdict (default 256, or 1024 with --think)')
    parser.add_argument('--limit', type=int, help='Only the first N predictions (for a quick test)')
    args = parser.parse_args()
    max_new_tokens = args.max_new_tokens or (1024 if args.think else 256)

    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        # Ranks finish at different times; the barrier only waits for the merge.
        dist.init_process_group('gloo', timeout=timedelta(hours=12))

    rows = load_rows(args.predictions)
    if args.limit:
        rows = rows[:args.limit]
    mine = rows[rank::world]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f'judgments_rank{rank}.jsonl'
    done = {json.loads(line)['id'] for line in shard.read_text().splitlines()} if shard.exists() else set()
    todo = [r for r in mine if r['id'] not in done]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': local}).eval()
    eos = tokenizer.eos_token_id

    with shard.open('a') as stream, torch.inference_mode():
        for row in tqdm(todo, desc=f'GPU {rank}', unit='q', position=rank, dynamic_ncols=True):
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': build_prompt(row)}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.think)
            inputs = tokenizer([prompt], return_tensors='pt').to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=eos)
            reply = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            correct, score, reason, raw = parse_verdict(reply)
            stream.write(json.dumps(dict(
                id=row['id'], question=row['question'], answer=row['answer'],
                prediction=row['prediction'], correct=correct, score=score, reason=reason,
                judge=args.model, parse_ok=correct is not None, judge_raw=raw[:2000]),
                ensure_ascii=False) + '\n')
            stream.flush()

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {}
        for worker in range(world):
            path = out / f'judgments_rank{worker}.jsonl'
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                record = json.loads(line)
                merged[record['id']] = record
        order = [r['id'] for r in rows if r['id'] in merged]
        (out / 'judgments.jsonl').write_text(
            ''.join(json.dumps(merged[i], ensure_ascii=False) + '\n' for i in order))
        graded = [merged[i] for i in order if merged[i]['correct'] is not None]
        scored = [g['score'] for g in graded if isinstance(g.get('score'), int)]
        summary = dict(
            predictions=len(rows), judged=len(order),
            parse_failures=sum(1 for i in order if merged[i]['correct'] is None),
            correct=sum(1 for g in graded if g['correct']),
            accuracy=round(sum(1 for g in graded if g['correct']) / len(graded), 4) if graded else None,
            mean_score=round(sum(scored) / len(scored), 4) if scored else None,
            judge=args.model, out=str(out / 'judgments.jsonl'))
        (out / 'judge_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
