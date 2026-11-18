#!/usr/bin/env bash
# Grounded-clip VideoSIAH pipeline: VideoJev-4B grounds each question's top-1 clip, then the
# answering model sees ONLY frames sampled inside that clip (vs. uniform whole-video sampling
# in eval_videosiah.sh). Everything else -- frame count, pixel cap, judge -- matches the
# baseline, so any delta is attributable to WHERE the frames come from.
#
# Stage 1 (grounding) is model-independent and cached in GROUND_OUT; it is reused if present.
# Stage 2 (answering) reuses scripts/eval_video_open.py with --grounding.
#
# Examples:
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct bash eval_videosiah_grounded.sh
#   REGROUND=1 bash eval_videosiah_grounded.sh    # force re-run stage 1
# Then grade with the strict LongVT judge:
#   RUN=runs/videosiah_<model-slug>_grounded_64f bash eval_videosiah_judge_longvt.sh
#
# Knobs: MODEL, NUM_FRAMES, MAX_PIXELS, MAX_NEW_TOKENS, THINK (1/0), NPROC, OUT, GROUND_OUT,
#        GROUND_FPS, GROUND_CONTEXT, THRESHOLD. NPROC picks the GPU count; set
#        CUDA_VISIBLE_DEVICES to choose which GPUs.
set -euo pipefail
source scripts/env.sh

QUESTIONS="${QUESTIONS:-data/benchmarks/videosiah/questions.json}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
NUM_FRAMES="${NUM_FRAMES:-4}"
MAX_PIXELS="${MAX_PIXELS:-50176}"
NPROC="${NPROC:-3}"

# ---- Stage 1: VideoJev-4B temporal grounding (cached & model-independent) ----
GROUND_OUT="${GROUND_OUT:-runs/videosiah_grounding_vj4b}"
GROUND_FILE="$GROUND_OUT/grounding.jsonl"
if [[ "${REGROUND:-0}" == "1" || ! -f "$GROUND_FILE" ]]; then
  echo "== stage 1: grounding -> $GROUND_FILE"
  torchrun --standalone --nproc-per-node="$NPROC" scripts/ground_videosiah.py \
    --questions "$QUESTIONS" \
    --checkpoint "${CHECKPOINT:-ZhuYuhan/VideoJev-4B}" \
    --fps "${GROUND_FPS:-4}" \
    --context "${GROUND_CONTEXT:-147456}" \
    --threshold "${THRESHOLD:-0.25}" \
    ${GROUND_LIMIT:+--limit "$GROUND_LIMIT"} \
    --out "$GROUND_OUT"
else
  echo "== stage 1: reusing existing $GROUND_FILE (set REGROUND=1 to redo)"
fi

# ---- Stage 2: answering model, frames restricted to each top-1 clip ----
if [[ -z "${THINK:-}" ]]; then
  if [[ "$MODEL" == *[Ii]nstruct* ]]; then THINK=0; else THINK=1; fi
fi
if [[ "$THINK" == "1" ]]; then
  THINK_FLAG=(--think); MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
else
  THINK_FLAG=();        MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
fi

SLUG="$(printf '%s' "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
OUT="${OUT:-runs/videosiah_${SLUG}_grounded_${NUM_FRAMES}f}"

echo "== stage 2: model=$MODEL frames=$NUM_FRAMES max_pixels=$MAX_PIXELS think=$THINK max_new_tokens=$MAX_NEW_TOKENS"
echo "   grounding=$GROUND_FILE  out=$OUT"

torchrun --standalone --nproc-per-node="$NPROC" scripts/eval_video_open.py \
  --questions "$QUESTIONS" \
  --model "$MODEL" \
  --num-frames "$NUM_FRAMES" \
  --max-pixels "$MAX_PIXELS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --grounding "$GROUND_FILE" \
  "${THINK_FLAG[@]}" \
  ${LIMIT:+--limit "$LIMIT"} \
  --out "$OUT"

echo "predictions -> $OUT/predictions.jsonl"
echo "grade with:  RUN=$OUT bash eval_videosiah_judge_longvt.sh"
