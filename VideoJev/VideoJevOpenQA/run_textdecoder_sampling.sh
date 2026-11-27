#!/usr/bin/env bash
# Run A: caption drafting on the first LIMIT VideoSIAH questions; predictions only (judge later
# with candidate_recall_videosiah.py).
#   1. VideoJev-4B grounds each question (top-1 Kadane interval) and caches the clip's vision features.
#   2. Qwen3.5-4B writes 1 greedy + SAMPLES sampled concise CAPTIONS of that clip (question-guided).
#   3. inspect.md / inspect.csv / candidates_pretty.json for manual review of all captions.
#   4. VideoJev-4B QA head (whole video, 2 FPS) picks ONE caption per question as the final answer
#      -> videojev_selection/predictions.jsonl (for the LLM judge) + final.md / final.csv.
# Steps 1, 2, 4 are reused if their outputs exist (REGROUND=1 / REDRAFT=1 / RESELECT=1 to redo).
#
# Examples (run from anywhere):
#   LIMIT=5 NPROC=1 bash VideoJevOpenQA/run_textdecoder_sampling.sh      # 5-question test on one GPU
#   LIMIT=100 NPROC=3 bash VideoJevOpenQA/run_textdecoder_sampling.sh     # 100 questions on 3 GPUs
#   LIMIT=all NPROC=3 VIDEO_FEATURES=/mnt/SSD2/videojev_cache/videosiah bash VideoJevOpenQA/run_textdecoder_sampling.sh
#   CUDA_VISIBLE_DEVICES=2 LIMIT=5 NPROC=1 bash VideoJevOpenQA/run_textdecoder_sampling.sh   # pick the GPU
# Knobs: MAX_FRAMES (default 512), MAX_PIXELS (default 50176 = 224*224, LongVT's per-frame cap), LIMIT (default 10; LIMIT=all for all 652 questions), NPROC (default 1), SAMPLES (default 8), CAPTION_MODE (caption |
#        caption_generic), QA_CONTEXT (default 98304), VIDEO_FEATURES, REGROUND, REDRAFT, RESELECT, OUT.
# VIDEO_FEATURES (default $OUT/grounding/video_features): whole-video VideoJev features cached by grounding
#   (~377 MB per video) and reused by selection, which then skips decoding. For many videos point it at
#   a big disk, e.g. VIDEO_FEATURES=/mnt/SSD2/videojev_cache/videosiah.
# REGROUND / REDRAFT / RESELECT=1 delete that stage's outputs and recompute them (stages resume by id).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/env.sh
HERE=VideoJevOpenQA
Q=data/benchmarks/videosiah/questions.json
LIMIT="${LIMIT:-10}"
[[ "$LIMIT" == "all" ]] && LIMIT=0   # 0 = every question (the Python stages treat --limit 0 as no limit)

NPROC="${NPROC:-1}"
SAMPLES="${SAMPLES:-8}"
CAPTION_MODE="${CAPTION_MODE:-caption}"
MAX_FRAMES="${MAX_FRAMES:-512}"   # frame cap per video for grounding + selection (LongVT eval: max_frames=512)
MAX_PIXELS="${MAX_PIXELS:-50176}" # per-frame pixel cap for grounding + selection (LongVT eval: max_pixels=50176)
if [[ "$LIMIT" == "0" ]]; then RUN_NAME=videosiah_caption_all_${MAX_FRAMES}f_${MAX_PIXELS}px; else RUN_NAME=videosiah_caption_limit${LIMIT}_${MAX_FRAMES}f_${MAX_PIXELS}px; fi
OUT="${OUT:-$HERE/runs/$RUN_NAME}"
G=$OUT/grounding
A=$OUT/${CAPTION_MODE}_top1_greedy+${SAMPLES}samples
S=$OUT/videojev_selection
VIDEO_FEATURES="${VIDEO_FEATURES:-$G/video_features}"
[[ "${REGROUND:-0}" == "1" ]] && rm -rf "$G" "$A" "$S" "$VIDEO_FEATURES"  # later stages depend on grounding
[[ "${REDRAFT:-0}" == "1" ]] && rm -rf "$A" "$S"
[[ "${RESELECT:-0}" == "1" ]] && rm -rf "$S"

if [[ ! -f $G/grounding.jsonl ]]; then
  echo "== grounding (VideoJev-4B, top-1 interval) -> $G  $(date)"
  # LongVT-matched input: MAX_FRAMES (512) uniform frames (every VideoSIAH video, 15-80 min, hits the cap at
  # 2 FPS) at <= MAX_PIXELS (224*224) per frame -- the same pixels LongVT's eval feeds Qwen2.5-VL.
  # VideoJev's 32x32-px tokens make that ~45 tokens per 2-frame group for 16:9, ~11.5k vision tokens per video.
  # The drafter gets the grounded clip's slice of these same features.
  torchrun --standalone --nproc-per-node=$NPROC scripts/ground_videosiah.py --questions $Q --limit $LIMIT \
    --checkpoint ZhuYuhan/VideoJev-4B --fps 2 --max-frames $MAX_FRAMES --max-pixels $MAX_PIXELS --context 98304 --threshold 0.25 \
    --clip-features $G/clip_features --video-features "$VIDEO_FEATURES" --top-k 1 --min-clip-groups 8 --max-clip-groups 32 --out $G
fi
if [[ ! -f $A/candidates.jsonl ]]; then
  echo "== captions (Qwen3.5-4B, $CAPTION_MODE, 1 greedy + $SAMPLES samples) -> $A  $(date)"
  torchrun --standalone --nproc-per-node=$NPROC $HERE/draft_videosiah.py --questions $Q --limit $LIMIT \
    --grounding $G/grounding.jsonl --clip-features $G/clip_features --model Qwen/Qwen3.5-4B \
    --mode $CAPTION_MODE --max-clips 1 --samples-per-clip $SAMPLES --out $A
fi
python $HERE/inspect_drafts_videosiah.py --grounding $G/grounding.jsonl --run "A=$A/candidates.jsonl" --out $OUT
if [[ ! -f $S/predictions.jsonl ]]; then
  echo "== VideoJev selection (QA head, whole video, cached features from grounding) -> $S  $(date)"
  # --fps/--max-frames only apply to videos missing from $VIDEO_FEATURES; cached videos reuse grounding's
  # frames (the same MAX_FRAMES uniform grid).
  torchrun --standalone --nproc-per-node=$NPROC $HERE/select_videosiah.py --questions $Q --limit $LIMIT \
    --candidates $A/candidates.jsonl --grounding $G/grounding.jsonl --checkpoint ZhuYuhan/VideoJev-4B \
    --video-features "$VIDEO_FEATURES" --fps 2 --max-frames $MAX_FRAMES --max-pixels $MAX_PIXELS --context "${QA_CONTEXT:-98304}" --out $S
fi
echo "== done $(date)"
echo "review:   $ROOT/$OUT/inspect.md"
echo "readable: $ROOT/$A/candidates_pretty.json"
echo "final:    $ROOT/$S/final.md   (judge input: $ROOT/$S/predictions.jsonl)"
