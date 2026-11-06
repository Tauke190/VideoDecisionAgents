#!/usr/bin/env bash
# Run from the repo root with the project venv active: bash eval_videosiah_judge.sh
# Grades the VideoSIAH open-ended predictions with a Qwen3 LLM judge (text-only).
# Reads $PRED (predictions.jsonl) and writes $OUT/judgments.jsonl + judge_summary.json.
# NPROC picks the GPU count; set CUDA_VISIBLE_DEVICES to choose which GPUs.
set -euo pipefail
source scripts/env.sh
RUN="${RUN:-runs/videosiah_qwen25vl7b_64f}"
PRED="${PRED:-$RUN/predictions.jsonl}"
# Fall back to the single-rank shard if the merged file is not there yet.
[[ -f "$PRED" ]] || PRED="$RUN/predictions_rank0.jsonl"
torchrun --standalone --nproc-per-node="${NPROC:-1}" scripts/judge_videosiah.py \
  --predictions "$PRED" \
  --model "${JUDGE_MODEL:-Qwen/Qwen3-8B}" \
  --out "${OUT:-$RUN/judge}"
