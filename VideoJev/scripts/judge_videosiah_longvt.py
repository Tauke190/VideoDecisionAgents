"""Grade VideoSIAH predictions with LongVT's exact scoring pipeline (for a like-for-like
comparison against the LongVT paper, arXiv 2511.20785).

Unlike judge_videosiah.py (a single lenient LLM pass that counts "same essential meaning"
as correct), this reproduces LongVT's `compute_score`
(custom_rewards/lmms_lab_recipe.py, wired into VideoSIAH-Eval via
lmms_eval_tasks/longvt/utils.py). For each prediction the pipeline is:

    1. relax_exact_match  -- deterministic string / substring / MCQ-letter match
    2. math_verify        -- symbolic equality (optional; skipped if math_verify is absent)
    3. strict 1/0 LLM judge, as a FALLBACK on items the matchers scored 0, and only
       when --use-judge is on. The judge can only flip 0 -> 1, never 1 -> 0.

The LLM judge prompt is LongVT's verbatim strict prompt ("output 1 for fully correct
answers and 0 for any other case; score 0 for partially correct answers or answers with
extra incorrect information"). It emits a bare 1 or 0.

Backends:
  --backend local   (default) grade with a local Qwen3 model via transformers, the same
                    model family as judge_videosiah.py, so the ONLY thing that changes
                    versus our lenient run is LongVT's pipeline + rubric.
  --backend openai  grade through an OpenAI-compatible endpoint, exactly like LongVT
                    (reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL_NAME), so you
                    can point at gpt-4o-mini or a served "judge" model for an exact match.

Writes one judgment per line and a judge_summary.json with the mean acc_score (the number
LongVT reports in Table 2), the composite score, and a breakdown of which stage decided
each item. Run under torchrun for multi-GPU sharding; the run resumes by ID and rank 0
merges and scores.
"""
import argparse
from datetime import timedelta
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# LongVT scoring helpers, ported verbatim from custom_rewards/lmms_lab_recipe.py
# (only reformatted/renamed where noted). Keep these faithful to the source.
# ---------------------------------------------------------------------------

# LongVT's verbatim strict judge prompts.
JUDGE_PROMPT = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.

# Input
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
1 or 0"""

JUDGE_PROMPT_WITH_ANSWER = """
You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case. You will receive the question, the ground truth answer, and the model prediction.

# Input
Question:
```
{question}
```

Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
1 or 0
"""


def extract_boxed_answer(predict_str: str) -> str:
    """Extract the answer from the last \\boxed{...} in the string (brace-balanced)."""
    boxed_start = "\\boxed{"
    start_indices = []
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1
    if not start_indices:
        return ""
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1
        while pos < len(predict_str):
            char = predict_str[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    content_start = start_pos + len(boxed_start)
                    results.append(predict_str[content_start:pos])
                    break
            pos += 1
    return results[-1] if results else ""


def extract_answer_tag(predict_str: str) -> str:
    """LongVT's extract_anwser_tag: <answer>...</answer>, then \\boxed{}, then a trailing number."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.search(pattern, predict_str)
    if match_result:
        return match_result.group(1)
    boxed_answer = extract_boxed_answer(predict_str)
    if boxed_answer:
        return boxed_answer
    lines = predict_str.strip().split('\n')
    for line in reversed(lines):
        if line.strip():
            number_match = re.search(r'\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)', line)
            if number_match:
                return number_match.group(1)
    return ""


def format_reward(predict_str: str) -> float:
    """LongVT's format reward: 1.0 for <think>..</think>..<answer>..</answer> or \\boxed{}."""
    think_answer_pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>", re.DOTALL)
    if re.fullmatch(think_answer_pattern, predict_str):
        return 1.0
    if extract_boxed_answer(predict_str):
        return 1.0
    if len(predict_str.strip()) > 50:
        has_math = bool(re.search(r'[=\+\-\*/\(\)\[\]\\]', predict_str))
        has_answer = bool(extract_answer_tag(predict_str))
        if has_math and has_answer:
            return 0.8
    return 0.0


def simple_parse(predict_str: str) -> str:
    if predict_str.endswith("."):
        predict_str = predict_str[:-1]
    return predict_str.strip()


def parse_mcq(predict_str: str) -> str:
    """LongVT's multiple-choice letter parser (A-H) across many formats."""
    if not predict_str or predict_str.strip() == "":
        return ""
    response = predict_str.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)
    response = " " + response + " "
    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append((choice, response.rfind(f"({choice})"), "parentheses"))
    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append((choice, response.rfind(f"{choice}."), "period"))
    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append((choice, response.rfind(f"{choice}:"), "colon"))
    for choice in all_choices:
        if f"{choice})" in response:
            candidates.append((choice, response.rfind(f"{choice})"), "right_paren"))
    for choice in all_choices:
        if f"{choice} " in response:
            candidates.append((choice, response.rfind(f"{choice} "), "space"))
    for choice in all_choices:
        if f"{choice}-" in response:
            candidates.append((choice, response.rfind(f"{choice}-"), "dash"))
    for choice in all_choices:
        if f"{choice}_" in response:
            candidates.append((choice, response.rfind(f"{choice}_"), "underscore"))
    for choice in all_choices:
        if f"{choice}=" in response:
            candidates.append((choice, response.rfind(f"{choice}="), "equals"))
    answer_phrases = [
        "the answer is", "answer is", "the correct answer is", "correct answer is",
        "the answer", "answer", "correct answer", "the correct answer",
        "the best answer is", "best answer is", "the best answer", "best answer",
        "the option is", "option is", "the correct option is", "correct option is",
        "the choice is", "choice is", "the correct choice is", "correct choice is",
        "i choose", "i select", "i pick", "my answer is", "my choice is",
    ]
    for phrase in answer_phrases:
        if phrase in response.lower():
            phrase_start = response.lower().find(phrase)
            for choice in all_choices:
                choice_pos = response.find(choice, phrase_start)
                if choice_pos != -1:
                    candidates.append((choice, choice_pos, "phrase"))
    for choice in all_choices:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, "start"))
    for choice in all_choices:
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))
    for i, choice in enumerate(all_choices):
        if f"{i+1}. {choice}" in response:
            candidates.append((choice, response.rfind(f"{i+1}. {choice}"), "numbered"))
    if not candidates:
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))
    if candidates:
        format_priority = {
            "start": 10, "end": 9, "numbered": 8, "phrase": 7,
            "parentheses": 6, "period": 5, "colon": 4, "right_paren": 3,
            "space": 2, "dash": 1, "underscore": 1, "equals": 1, "fallback": 0,
        }
        candidates.sort(key=lambda x: (format_priority[x[2]], -x[1]), reverse=True)
        return candidates[0][0]
    return ""


def relax_exact_match(predict_str: str, ground_truth: str, relax_portion: float = 0.9) -> float:
    """LongVT's deterministic matcher: MCQ letter, 0.9-length substring containment, or exact."""
    if ground_truth in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        predict_str = parse_mcq(predict_str)
        return 1.0 if predict_str == ground_truth else 0.0
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0


# Optional symbolic-math check (LongVT uses math_verify; it is not installed here, and it
# effectively never fires on VideoSIAH's free-form text answers, so we skip it when absent).
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify  # type: ignore
    HAVE_MATH_VERIFY = True
except Exception:
    HAVE_MATH_VERIFY = False


def build_judge_prompt(predict_str, ground_truth, question):
    if question:
        return JUDGE_PROMPT_WITH_ANSWER.format(question=question, answer=ground_truth, prediction=predict_str)
    return JUDGE_PROMPT.format(answer=ground_truth, prediction=predict_str)


def parse_binary(reply: str) -> int:
    """LongVT does int(reply) else 0; we tolerate a leading token by taking the first 0/1."""
    reply = reply.strip()
    try:
        return 1 if int(reply) == 1 else 0
    except Exception:
        match = re.search(r'[01]', reply)
        return int(match.group()) if match else 0


def score_row(row, grade, use_judge):
    """Reproduce LongVT compute_score(data_source='longvt') for one prediction.

    `grade(predict_str, ground_truth, question) -> (int score, str raw)` runs the LLM judge.
    Returns a dict with acc_score (0/1), format_reward_score, composite score, the extracted
    answer, which stage decided it, and the judge's raw reply (if the judge ran).
    """
    solution_str = str(row['prediction']).strip()
    ground_truth = str(row['answer'])
    question = row.get('question', '')

    format_score_const = 0.1
    format_reward_score = format_reward(solution_str)
    predict_str = simple_parse(extract_answer_tag(solution_str).strip())
    gt = simple_parse(ground_truth)

    stage, judge_raw, judge_ran = None, '', False
    acc_score = relax_exact_match(predict_str, gt)
    if acc_score:
        stage = 'string'

    if acc_score == 0.0 and HAVE_MATH_VERIFY:
        try:
            acc_score = float(int(_mv_verify(_mv_parse(gt), _mv_parse(predict_str))))
            if acc_score:
                stage = 'math'
        except Exception:
            acc_score = 0.0

    # LLM fallback #1: judge the extracted answer span. Skipped when the span is empty
    # (judging "" is pointless and LongVT would score it 0 anyway), which keeps results
    # identical while avoiding a wasted judge call -- our untagged predictions extract to "".
    if acc_score == 0.0 and use_judge and predict_str:
        verdict, judge_raw = grade(predict_str, gt, question)
        judge_ran = True
        acc_score = float(verdict)
        if acc_score:
            stage = 'llm_extracted'

    # LLM fallback #2 (LongVT): judge the raw generation when no format and it is short.
    if acc_score == 0.0 and use_judge and format_reward_score == 0.0 and len(solution_str) < 500:
        verdict, judge_raw = grade(solution_str, gt, question)
        judge_ran = True
        acc_score = float(verdict)
        if acc_score:
            stage = 'llm_raw'

    if stage is None:  # acc stayed 0: did a judge see it, or was it never evaluated?
        stage = 'judge_wrong' if judge_ran else 'unjudged'
    score = (1.0 - format_score_const) * acc_score + format_score_const * format_reward_score
    return dict(acc_score=acc_score, format_reward_score=format_reward_score, score=score,
                predict_str=predict_str, stage=stage, judge_ran=judge_ran, judge_raw=judge_raw[:2000])


def load_rows(path):
    rows, seen = [], set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record['id'] in seen:
            continue
        seen.add(record['id'])
        for field in ('id', 'question', 'answer', 'prediction'):
            if field not in record:
                raise KeyError(f'Prediction {record.get("id", "?")} is missing "{field}"')
        rows.append(record)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--out', required=True, help='Output directory for judgments.jsonl')
    parser.add_argument('--backend', choices=['local', 'openai'], default='local')
    parser.add_argument('--model', default='Qwen/Qwen3-8B',
                        help='local backend: HF model id for the judge')
    parser.add_argument('--no-judge', dest='use_judge', action='store_false',
                        help='Only run the deterministic string/math matchers (LongVT with USE_LLM_JUDGE off)')
    parser.add_argument('--max-new-tokens', type=int, default=8,
                        help='local backend decode budget for the 1/0 verdict')
    parser.add_argument('--limit', type=int, help='Only the first N predictions (for a quick test)')
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from tqdm import tqdm

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
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

    # Build the grade() callable for the chosen backend.
    if args.use_judge and args.backend == 'local':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.cuda.set_device(local)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation='flash_attention_2',
            device_map={'': local}).eval()
        eos = tokenizer.eos_token_id

        @torch.inference_mode()
        def grade(predict_str, ground_truth, question):
            prompt = build_judge_prompt(predict_str, ground_truth, question)
            messages = [{'role': 'user', 'content': prompt}]  # LongVT uses a bare user turn
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer([text], return_tensors='pt').to(model.device)
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, pad_token_id=eos)
            reply = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return parse_binary(reply), reply
    elif args.use_judge and args.backend == 'openai':
        from openai import OpenAI  # imported lazily; only needed for this backend
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
                        base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
        model_name = args.model if args.model != 'Qwen/Qwen3-8B' else os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')

        def grade(predict_str, ground_truth, question):
            prompt = build_judge_prompt(predict_str, ground_truth, question)
            response = client.chat.completions.create(
                model=model_name, max_tokens=5,
                messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}])
            reply = response.choices[0].message.content or ''
            return parse_binary(reply), reply
    else:
        def grade(predict_str, ground_truth, question):  # matchers only
            return 0, ''

    with shard.open('a') as stream:
        for row in tqdm(todo, desc=f'rank {rank}', unit='q', position=rank, dynamic_ncols=True):
            result = score_row(row, grade, args.use_judge)
            stream.write(json.dumps(dict(
                id=row['id'], question=row['question'], answer=row['answer'],
                prediction=row['prediction'], **result), ensure_ascii=False) + '\n')
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
        graded = [merged[i] for i in order]
        stages = {}
        for g in graded:
            stages[g['stage']] = stages.get(g['stage'], 0) + 1
        n = len(graded)
        summary = dict(
            predictions=len(rows), judged=n,
            correct=sum(1 for g in graded if g['acc_score'] >= 1.0),
            acc_score=round(sum(g['acc_score'] for g in graded) / n, 4) if n else None,
            composite_score=round(sum(g['score'] for g in graded) / n, 4) if n else None,
            format_score=round(sum(g['format_reward_score'] for g in graded) / n, 4) if n else None,
            judge_calls=sum(1 for g in graded if g.get('judge_ran')),
            decided_by=stages,
            backend=args.backend, judge=(args.model if args.use_judge else None),
            use_judge=args.use_judge, math_verify=HAVE_MATH_VERIFY,
            out=str(out / 'judgments.jsonl'))
        (out / 'judge_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
