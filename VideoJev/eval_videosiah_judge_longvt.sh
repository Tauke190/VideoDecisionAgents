#!/usr/bin/env bash
# Run from the repo root with the project venv active: bash eval_videosiah_judge_longvt.sh
# Grades VideoSIAH open-ended predictions with LongVT's exact scoring pipeline
# (relax_exact_match -> math_verify -> strict 1/0 LLM judge as a fallback), for a
# like-for-like comparison with the LongVT paper (arXiv 2511.20785, Table 2).
# Reads $PRED (predictions.jsonl) and writes $OUT/judgments.jsonl + judge_summary.json.
# NPROC picks the GPU count (local backend); set CUDA_VISIBLE_DEVICES to choose which GPUs.
#
# BACKEND=local (default): judge with a local Qwen3 model (same model as the lenient judge,
#   so only LongVT's pipeline + rubric differ).
# BACKEND=openai: judge through an OpenAI-compatible endpoint, exactly like LongVT. Set
#   OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL_NAME (or pass JUDGE_MODEL) and NPROC=1.
set -euo pipefail
source scripts/env.sh
RUN="${RUN:-runs/videosiah_qwen25vl7b_64f}"
PRED="${PRED:-$RUN/predictions.jsonl}"
[[ -f "$PRED" ]] || PRED="$RUN/predictions_rank0.jsonl"
torchrun --standalone --nproc-per-node="${NPROC:-1}" scripts/judge_videosiah_longvt.py \
  --predictions "$PRED" \
  --backend "${BACKEND:-local}" \
  --model "${JUDGE_MODEL:-Qwen/Qwen3-8B}" \
  --out "${OUT:-$RUN/judge_longvt}"
