#!/usr/bin/env bash
# Run from the repo root with the project venv active: bash eval_longvideobench.sh
# NPROC=3 picks the GPU count; set CUDA_VISIBLE_DEVICES to choose which GPUs.
# QA context is lowered to 64K to fit a 24 GB GPU (official: 262144), so scores are not official.
set -euo pipefail
source scripts/env.sh
torchrun --standalone --nproc-per-node="${NPROC:-3}" scripts/evaluate_hf.py \
  --checkpoint ZhuYuhan/VideoJev-4B \
  --benchmark longvideobench --data-root data/benchmarks --out "${OUT:-runs/longvideobench_64k}" \
  --qa-context 65536
