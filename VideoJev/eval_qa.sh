#!/usr/bin/env bash
# Run from the repo root with the project venv active: bash eval.sh
# NPROC=3 picks the GPU count; set CUDA_VISIBLE_DEVICES to choose which GPUs.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}" \
source scripts/env.sh
torchrun --standalone --nproc-per-node="${NPROC:-2}" scripts/evaluate_hf.py \
  --checkpoint ZhuYuhan/VideoJev-4B \
  --benchmark lvbench --data-root data/benchmarks --out "${OUT:-runs/benchmarks_64k}" \
  --qa-context 65536