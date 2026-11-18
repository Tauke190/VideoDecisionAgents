#!/usr/bin/env bash
# One common runner for VideoSIAH-Eval predictions -- swap the model with MODEL= only.
# The evaluation protocol is fixed to the matched-LongVT sparse setting:
#   64 frames, per-frame max_pixels=50176 (224x224). Model loads via Auto* classes, so
#   Qwen2.5-VL, Qwen3-VL and Qwen3.5-Moe (incl. quantized) all just work.
#
# Examples:
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct       bash eval_videosiah.sh
#   MODEL=Qwen/Qwen3-VL-4B-Thinking         bash eval_videosiah.sh
#   MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4    bash eval_videosiah.sh   # needs a GPTQ backend installed
# Then grade with the strict LongVT judge:
#   RUN=runs/videosiah_<model-slug>_64f bash eval_videosiah_judge_longvt.sh
#
# Knobs (all overridable): NUM_FRAMES, MAX_PIXELS, MAX_NEW_TOKENS, THINK (1/0), NPROC, OUT.
# NPROC picks the GPU count; set CUDA_VISIBLE_DEVICES to choose which GPUs.
set -euo pipefail
source scripts/env.sh

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"
NUM_FRAMES="${NUM_FRAMES:-64}"
MAX_PIXELS="${MAX_PIXELS:-50176}"

# Thinking checkpoints emit <think>...</think> before <answer>; they need room to decode and
# the <think>/<answer> system prompt so the judge can extract the answer. Default: think unless
# the model name says "Instruct". Override with THINK=1 or THINK=0.
if [[ -z "${THINK:-}" ]]; then
  if [[ "$MODEL" == *[Ii]nstruct* ]]; then THINK=0; else THINK=1; fi
fi
if [[ "$THINK" == "1" ]]; then
  # Thinking traces on hard long-video questions routinely exceed 2048 and get truncated before
  # </think> (empty answer -> auto-0). 8192 catches almost all; raise further if truncation remains.
  THINK_FLAG=(--think); MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
else
  THINK_FLAG=();        MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
fi

SLUG="$(printf '%s' "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
OUT="${OUT:-runs/videosiah_${SLUG}_${NUM_FRAMES}f}"

echo "model=$MODEL  frames=$NUM_FRAMES  max_pixels=$MAX_PIXELS  think=$THINK  max_new_tokens=$MAX_NEW_TOKENS"
echo "out=$OUT"

torchrun --standalone --nproc-per-node="${NPROC:-3}" scripts/eval_video_open.py \
  --questions data/benchmarks/videosiah/questions.json \
  --model "$MODEL" \
  --num-frames "$NUM_FRAMES" \
  --max-pixels "$MAX_PIXELS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${THINK_FLAG[@]}" \
  --out "$OUT"

echo "predictions -> $OUT/predictions.jsonl"
echo "grade with:  RUN=$OUT bash eval_videosiah_judge_longvt.sh"
