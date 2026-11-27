"""Compare full-video coarse scans at different frame budgets on CG-Bench (System 1 only, no zoom).

Usage: python VideoJevOpenQA/compare_cgbench_scans.py VideoJevOpenQA/runs/cgbench_scan100_f64 ..._f128 ..._f256
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_cgbench_grounding import metrics  # noqa: E402

BINS = [('<0.6', 0, .6), ('0.6-0.8', .6, .8), ('0.8-0.95', .8, .95), ('>=0.95', .95, 2)]


def main():
    runs = [Path(p) for p in sys.argv[1:]]
    table = ['| scan | cell (s) | main peak hit | + 1 peak | any peak | clue recall | IoU | window (s) | confident | s / q |',
             '|---|---|---|---|---|---|---|---|---|---|']
    calib = ['| scan | ' + ' | '.join(b[0] for b in BINS) + ' |', '|---|' + '---|' * len(BINS)]
    for run in runs:
        P = [json.loads(l) for l in (run / 'predictions.jsonl').read_text().splitlines()]
        n = len(P)
        L0 = [p['levels'][0] for p in P]
        main = [metrics([l['window']], p['spans']) for p, l in zip(P, L0)]
        plus1 = [metrics([l['window'], *[x['window'] for x in l['peaks'][:1]]], p['spans']) for p, l in zip(P, L0)]
        anyp = [metrics([l['window'], *[x['window'] for x in l['peaks']]], p['spans']) for p, l in zip(P, L0)]
        mean = lambda xs, k: sum(x[k] for x in xs) / n
        secs = sum(l['decode_s'] + l['ground_s'] for l in L0) / n
        table.append(f'| {run.name} ({n}) | {st.median(l["cell"] for l in L0):.0f} | {mean(main, "hit"):.2f} | '
                     f'{mean(plus1, "hit"):.2f} | {mean(anyp, "hit"):.2f} | {mean(main, "recall"):.2f} | '
                     f'{mean(main, "iou"):.3f} | {st.median(m["length"] for m in main):.0f} | '
                     f'{sum(l["confident"] for l in L0)} | {secs:.1f} |')
        cells = []
        for name, lo, hi in BINS:
            h = [m['hit'] for m, l in zip(main, L0) if lo <= l['confidence'] < hi]
            cells.append(f'{sum(h) / len(h):.2f} (n={len(h)})' if h else '-')
        calib.append(f'| {run.name} | ' + ' | '.join(cells) + ' |')
    out = ['# Full-video coarse scans on CG-Bench\n', *table, '',
           '## Calibration: main-peak hit rate by confidence', '', *calib]
    print('\n'.join(out))
    (runs[-1].parent / 'cgbench_scan_compare.md').write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
