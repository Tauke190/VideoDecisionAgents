source scripts/env.sh
torchrun --standalone --nproc-per-node="${NPROC:-1}" scripts/evaluate_hf.py \
  --checkpoint ZhuYuhan/VideoJev-4B \
  --benchmark charades --data-root data/benchmarks --out "${OUT:-runs/charades}"