"""Raw VideoJev-4B grounding scores for one VideoSIAH question, with NO post-processing.

Samples --max-frames frames (uniform over the whole video when the video is long), runs the
grounding head once and dumps the model's per-cell output logits exactly as returned
(output.logits[:score_mask.sum()]): no sigmoid, no threshold shift, no Kadane, no rounding.
One cell = one temporal group = 2 consecutive sampled frames, so 64 frames -> 32 scores.

Example:
  python VideoJevOpenQA/raw_grounding_scores.py --id videosiah/0008 --max-frames 64
Writes <out>/<id>_<frames>f.json and prints a table (cell, time span, raw logit).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from videojev_runtime import patch_decode_cap, videojev_modules  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', default='data/benchmarks/videosiah/questions.json')
    parser.add_argument('--id', default='videosiah/0008')
    parser.add_argument('--query', help='Override the grounding query (default: the VideoSIAH question)')
    parser.add_argument('--checkpoint', default='ZhuYuhan/VideoJev-4B')
    parser.add_argument('--max-frames', type=int, default=64)
    parser.add_argument('--fps', type=float, default=4.0, help='Upper bound; long videos hit --max-frames first')
    parser.add_argument('--context', type=int, default=98304)
    parser.add_argument('--decode-max-edge', type=int, default=1280,
                        help='Decode-time downscale (memory only; the model input size is set by the processor)')
    parser.add_argument('--out', default='VideoJevOpenQA/runs/raw_grounding')
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoProcessor

    source = Path(args.questions)
    row = next(r for r in json.loads(source.read_text()) if r['id'] == args.id)
    path = str((source.parent / row['video_path']).resolve())
    query = args.query or row['question']

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
                                      device_map={'': 0}, attn_implementation='flash_attention_2')
    model.prepare_for_inference()
    patch_decode_cap(model, args.decode_max_edge)
    processing = videojev_modules(model)['processing_videojev']

    with torch.inference_mode():
        pixels, record = processing.prepare_video(processor, path, fps=args.fps, max_frames=args.max_frames,
                                                  context=args.context)
        features = model.encode_video(pixels, torch.tensor([record['grid_thw']]))
        del pixels
        questions = [dict(id=row['id'], task='grounding', query=query)]
        (batch, rows), = list(processing.prepare_query_groups(processor, record, questions, args.context))
        output = model(**batch, visual_features=features)
        count = int(output.score_mask[0].sum())
        logits = output.logits[0, :count].float().cpu().tolist()

    cells = [dict(cell=i, start=round(a, 3), end=round(b, 3), time=round(t, 3), logit=l)
             for i, ((a, b), t, l) in enumerate(zip(record['score_bounds'], record['score_times'], logits))]
    frames = record['frames']
    result = dict(id=row['id'], video_path=path, query=query, answer=row['answer'],
                  num_frames=len(frames) if isinstance(frames, list) else frames, grid_thw=record['grid_thw'],
                  duration=record.get('duration'), context=args.context, num_cells=count,
                  note='raw output.logits per cell; no sigmoid / threshold / Kadane / rounding', cells=cells)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{row["id"].replace("/", "_")}_{result["num_frames"]}f.json'
    target.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')

    print(f'{row["id"]}  frames={result["num_frames"]}  cells={count}  grid={record["grid_thw"]}')
    print(f'Q: {query}\nGT: {row["answer"]}\n')
    print(f'{"cell":>4}  {"start(s)":>9}  {"end(s)":>9}  {"raw logit":>10}')
    for c in cells:
        print(f'{c["cell"]:>4}  {c["start"]:>9.1f}  {c["end"]:>9.1f}  {c["logit"]:>10.4f}')
    print(f'\nsaved: {target.resolve()}')


if __name__ == '__main__':
    main()
