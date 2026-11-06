"""Download CG-Bench (gated HF dataset) and lay it out like the Charades grounding benchmark.

Videos and annotations live under --root (big storage); the repo directory --out gets
questions.jsonl plus symlinks, exactly as data/benchmarks/charades does. Each question's
clue intervals become its grounding spans; the question text is the query.
"""
import argparse
import json
import zipfile
from pathlib import Path

REPO = 'CG-Bench/CG-Bench'
VIDEO_CHUNKS = 69


def download(root, chunks):
    from huggingface_hub import hf_hub_download
    for name in ('cgbench.json', 'cgbench_mini.json', 'README.md'):
        if not (root/name).is_file():
            hf_hub_download(REPO, name, repo_type='dataset', local_dir=root)
    videos, done = root/'videos', root/'chunks_done'
    videos.mkdir(exist_ok=True)
    done.mkdir(exist_ok=True)
    for index in chunks:
        name = f'video_chunk_{index:02d}.zip'
        if (done/name).exists():
            continue
        archive = Path(hf_hub_download(REPO, name, repo_type='dataset', local_dir=root/'zips'))
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                # Flatten and keep only videos so archive paths cannot escape videos/.
                target = videos/Path(member.filename).name
                if member.is_dir() or target.suffix != '.mp4':
                    continue
                with z.open(member) as src, open(target.with_suffix('.part'), 'wb') as dst:
                    while block := src.read(1 << 24):
                        dst.write(block)
                target.with_suffix('.part').rename(target)
        archive.unlink()
        (done/name).touch()
        print(json.dumps(dict(event='chunk_done', chunk=name)), flush=True)


def convert(root, out, annotations):
    raw = json.loads((root/annotations).read_text())
    rows, official = [], {}
    for item in sorted(raw, key=lambda x: (x['video_uid'], x['qid'])):
        vid, duration = item['video_uid'], float(item['duration'])
        spans = [[float(s), float(e)] for s, e in item['clue_intervals']]
        if not spans or any(not 0 <= s < e <= duration for s, e in spans):
            raise ValueError(f'Bad clue intervals for qid {item["qid"]}')
        rows.append(dict(id=f'cgbench/{vid}/{item["qid"]}', task='grounding', video_key=f'cgbench/{vid}',
                         video_path=f'videos/{vid}.mp4', duration=duration, query=item['question'],
                         spans=spans, source='CG-Bench', split='test',
                         domain=item['domain'], sub_category=item['sub_category']))
        entry = official.setdefault(vid, dict(duration=duration, spans=[], queries=[]))
        entry['spans'].append(spans)
        entry['queries'].append(item['question'])
    missing = sorted({r['video_path'] for r in rows if not (root/r['video_path']).is_file()})
    if missing:
        raise FileNotFoundError(f'{len(missing)} videos missing under {root}, e.g. {missing[:3]}')
    (root/'official.json').write_text(json.dumps(official, indent=4))
    out.mkdir(parents=True, exist_ok=True)
    with open(out/'questions.jsonl', 'w') as f:
        f.writelines(json.dumps(r, separators=(',', ':')) + '\n' for r in rows)
    for link, target in (('videos', root/'videos'), ('official.json', root/'official.json')):
        if not (out/link).is_symlink():
            (out/link).symlink_to(target)
    print(json.dumps(dict(event='converted', questions=len(rows), videos=len(official), out=str(out))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/SSD2/CG-Bench', help='Storage for videos and raw annotations')
    parser.add_argument('--out', default='data/benchmarks/cgbench')
    parser.add_argument('--annotations', default='cgbench.json', choices=('cgbench.json', 'cgbench_mini.json'))
    parser.add_argument('--chunks', type=int, nargs='*', default=list(range(1, VIDEO_CHUNKS + 1)))
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download(root, args.chunks)
    convert(root, Path(args.out), args.annotations)
