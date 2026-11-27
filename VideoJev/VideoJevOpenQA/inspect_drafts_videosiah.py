"""Side-by-side manual-inspection sheet for draft runs (no judging).

Joins grounding.jsonl (VideoJev's top-K intervals) with one or more candidates.jsonl runs and
writes, per question: the question, the GT answer, the video file, every grounded interval as
mm:ss (so you can scrub to it and check whether VideoJev's window actually holds the answer --
VideoSIAH has no GT timestamps), and each run's candidates labelled with the interval they
came from. Outputs inspect.md (readable) and inspect.csv (one row per candidate, with empty
`correct` and `window_ok` columns to fill in by hand).

    python VideoJevOpenQA/inspect_drafts_videosiah.py --grounding G/grounding.jsonl \
        --run A=runs/.../candidates.jsonl --run B=runs/.../candidates.jsonl --out DIR
"""
import argparse
import csv
import json
from pathlib import Path


def mmss(seconds):
    seconds = int(round(seconds))
    return f'{seconds // 60:d}:{seconds % 60:02d}'


def span(interval):
    return f'{mmss(interval[0])}–{mmss(interval[1])}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grounding', required=True)
    parser.add_argument('--run', action='append', required=True, help='NAME=path/to/candidates.jsonl (repeatable)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    grounding = {json.loads(l)['id']: json.loads(l) for l in Path(args.grounding).read_text().splitlines() if l.strip()}
    runs = {}
    for spec in args.run:
        name, path = spec.split('=', 1)
        runs[name] = {json.loads(l)['id']: json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()}
        # Readable copy next to the run: one entry per candidate with the interval it came from.
        pretty = [dict(id=r['id'], question=r['question'], answer=r['answer'], mode=r.get('mode', 'answer'),
                       candidates=[dict(n=i + 1, text=c, interval_rank=s['clip_rank'] + 1, kind=s['kind'],
                                        interval=s['interval'], score=s['score'], clip_span=s['clip_span'])
                                   for i, (c, s) in enumerate(zip(r['candidates'], r['sources']))])
                  for r in runs[name].values()]
        (Path(path).parent / 'candidates_pretty.json').write_text(json.dumps(pretty, indent=2, ensure_ascii=False) + '\n')
    ids = [i for i in grounding if all(i in r for r in runs.values())]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f'# Draft inspection: {", ".join(runs)} ({len(ids)} questions)\n',
          'Intervals are VideoJev predictions (NOT ground truth). `score` = Kadane log-likelihood gain '
          '(<= 0: nothing above threshold left), `peak` = max per-cell probability.\n']
    rows = []
    for n, qid in enumerate(ids, 1):
        g = grounding[qid]
        first = next(iter(runs.values()))[qid]
        md += [f'\n---\n\n## {n}. `{qid}` — `{Path(g["video_path"]).name}`\n',
               f'**Q:** {first["question"]}\n', f'**GT:** {first["answer"]}\n',
               '| # | VideoJev interval | score | peak |', '|---|---|---|---|']
        used = 1 + max(s['clip_rank'] for run in runs.values() for s in run[qid].get('sources') or [dict(clip_rank=0)])
        intervals = g.get('intervals') or [dict(interval=g['interval'], score=None, peak=g.get('peak'))]
        for k, it in enumerate(intervals[:used], 1):  # only intervals some run actually drafted from
            md.append(f'| {k} | {span(it["interval"])} ({it["interval"][0]:.0f}–{it["interval"][1]:.0f}s) '
                      f'| {it["score"]} | {it["peak"]} |')
        for name, run in runs.items():
            rec = run[qid]
            md.append(f'\n**Run {name}** — {rec.get("mode", "answer")} mode ({len(rec["candidates"])} candidates)\n')
            for j, (cand, src) in enumerate(zip(rec['candidates'], rec.get('sources') or [{}] * len(rec['candidates'])), 1):
                where = (f'interval #{src["clip_rank"] + 1} {span(src["interval"])}, {src["kind"]}'
                         if src else '')
                md.append(f'{j}. {cand}  <sub>({where})</sub>')
                rows.append(dict(id=qid, video=Path(g['video_path']).name, question=first['question'],
                                 answer=first['answer'], run=name, candidate_index=j, candidate=cand,
                                 interval_rank=src.get('clip_rank', 0) + 1 if src else '',
                                 interval=span(src['interval']) if src else '', kind=src.get('kind', ''),
                                 correct='', window_ok=''))
    (out / 'inspect.md').write_text('\n'.join(md) + '\n')
    with (out / 'inspect.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(questions=len(ids), candidates=len(rows), md=str(out / 'inspect.md'),
                          csv=str(out / 'inspect.csv'))))


if __name__ == '__main__':
    main()
