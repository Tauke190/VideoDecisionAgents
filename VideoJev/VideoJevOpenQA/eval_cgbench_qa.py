"""CG-Bench QA metrics for hierarchical sampling (official protocol, arXiv 2412.12075).

long-acc  multiple-choice accuracy (Qwen answers from the top-K leaves)
mIoU      tIoU of the predicted intervals (top-K leaf windows) vs the clue intervals
rec@IoU   mean over tau in {0.1..0.5} of P(tIoU > tau)
acc@IoU   mean over tau in {0.1..0.5} of P(correct and tIoU > tau)
OE-acc    open-ended accuracy from the local LongVT judge (not the official GPT-4o judge)

Usage: python VideoJevOpenQA/eval_cgbench_qa.py VideoJevOpenQA/runs/cgbench_qa_s100
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

TAUS = [0.1, 0.2, 0.3, 0.4, 0.5]


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def metrics(windows, spans):
    # merged intersection / union = the paper's multi-interval tIoU
    windows, spans = merge(windows), merge(spans)
    inter = sum(max(0.0, min(b, d) - max(a, c)) for a, b in windows for c, d in spans)
    wlen, slen = sum(b - a for a, b in windows), sum(b - a for a, b in spans)
    return dict(hit=float(inter > 0), iou=inter / (wlen + slen - inter))


def main():
    run = Path(sys.argv[1])
    P = [json.loads(l) for l in (run / 'predictions.jsonl').read_text().splitlines()]
    n = len(P)
    iou = [metrics(p['answer_windows'], p['spans'])['iou'] for p in P]
    hit = [metrics(p['answer_windows'], p['spans'])['hit'] for p in P]
    correct = [bool(p.get('correct')) for p in P]
    res = dict(
        questions=n,
        long_acc=sum(correct) / n,
        miou=sum(iou) / n,
        rec_iou=sum(sum(i > t for i in iou) / n for t in TAUS) / len(TAUS),
        acc_iou=sum(sum(c and i > t for c, i in zip(correct, iou)) / n for t in TAUS) / len(TAUS),
        clue_hit=sum(hit) / n,
        acc_given_hit=sum(c for c, h in zip(correct, hit) if h) / max(1, sum(hit)),
        acc_given_miss=sum(c for c, h in zip(correct, hit) if not h) / max(1, n - sum(hit)),
        unparsed=sum(p.get('pred_letter') is None for p in P),
        mean_windows=sum(len(p['answer_windows']) for p in P) / n,
        mean_grounding_calls=sum(p['num_levels'] for p in P) / n)
    judged = run / 'open' / 'judge_longvt' / 'judgments.jsonl'
    if judged.exists():
        J = [json.loads(l) for l in judged.read_text().splitlines()]
        res['oe_acc'] = sum(float(j['acc_score']) for j in J) / len(J)
        res['oe_questions'] = len(J)
    cats = defaultdict(list)
    for p, c in zip(P, correct):
        cats[p.get('sub_category')].append(c)

    pct = lambda x: f'{100 * x:.1f}'
    md = [f'# CG-Bench QA: {run.name} ({n} questions)\n', '| metric | value |', '|---|---|',
          f'| long-acc (MCQ, %) | {pct(res["long_acc"])} |', f'| mIoU (%) | {pct(res["miou"])} |',
          f'| rec@IoU (%) | {pct(res["rec_iou"])} |', f'| acc@IoU (%) | {pct(res["acc_iou"])} |']
    if 'oe_acc' in res:
        md.append(f'| OE-acc (Qwen3-8B judge, %) | {pct(res["oe_acc"])} |')
    md += ['', '| diagnostic | value |', '|---|---|',
           f'| leaves touch the clue (%) | {pct(res["clue_hit"])} |',
           f'| MCQ acc when the clue is touched (%) | {pct(res["acc_given_hit"])} |',
           f'| MCQ acc when the clue is missed (%) | {pct(res["acc_given_miss"])} |',
           f'| unparsed answers | {res["unparsed"]} |',
           f'| mean windows to Qwen | {res["mean_windows"]:.2f} |',
           f'| mean grounding calls | {res["mean_grounding_calls"]:.2f} |',
           '', '| sub-category | n | long-acc (%) |', '|---|---|---|']
    md += [f'| {k} | {len(v)} | {pct(sum(v) / len(v))} |' for k, v in sorted(cats.items(), key=lambda x: -len(x[1]))]
    print('\n'.join(md))
    (run / 'cgbench_qa.md').write_text('\n'.join(md) + '\n')
    (run / 'cgbench_qa.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
