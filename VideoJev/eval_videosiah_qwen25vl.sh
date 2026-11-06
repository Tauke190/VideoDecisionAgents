#!/usr/bin/env bash
# Run from the repo root with the project venv active: bash eval_videosiah_qwen25vl.sh
# Qwen2.5-VL-7B on VideoSIAH-Eval, sparse mode: 64 frames sampled uniformly over each whole video.
# Writes predictions only (no scoring) to $OUT/predictions.jsonl; re-running resumes by question ID.
# Videos that fail to decode are skipped and listed in $OUT/skipped_rank*.jsonl.
# NPROC=3 picks the GPU count; set CUDA_VISIBLE_DEVICES to choose which GPUs.
set -euo pipefail
source scripts/env.sh
torchrun --standalone --nproc-per-node="${NPROC:-3}" scripts/eval_qwen25vl_open.py \
  --questions data/benchmarks/videosiah/questions.json \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --num-frames 64 \
  --out "${OUT:-runs/videosiah_qwen25vl7b_64f}"