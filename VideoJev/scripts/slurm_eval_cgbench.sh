#!/bin/bash
#SBATCH --job-name=cgbench_videojev
#SBATCH --output=/home/av354855/projects/VideoJev/runs/slurm/cgbench_eval-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=90G               # decoding up to 2048 frames of ~28 min videos per query
# Create the log directory once before submitting: mkdir -p runs/slurm
# Submit: sbatch scripts/slurm_eval_cgbench.sh   (MINI=1 for the 3,000-question subset)

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-videojev}"

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/projects/VideoJev
source scripts/env.sh
set -euo pipefail

name=cgbench
[ "${MINI:-0}" != 1 ] || name=cgbench_mini
# Directory holding questions.jsonl and videos/ (relative video paths resolve against it).
cgbench_dir="${CGBENCH_DIR:-data/benchmarks/${name}}"
checkpoint="${CHECKPOINT:-ZhuYuhan/VideoJev-4B}"
out="${OUT:-runs/${name}-${SLURM_JOB_ID}}"

# Same CUDA 12.8 ptxas check as the other cluster recipes; SKIP_PTXAS_CHECK=1 to bypass.
[ "${SKIP_PTXAS_CHECK:-0}" = 1 ] || source scripts/cluster_cuda128.sh

echo "=== VideoJev | ${name} | ${checkpoint} | out=${out} ==="
# Validate every row and video path before loading weights.
python -B scripts/evaluate_hf.py --checkpoint "${checkpoint}" --out "${out}" \
    --benchmark custom --manifest "${cgbench_dir}/questions.jsonl" --dry-run

torchrun --standalone --nproc-per-node=1 scripts/evaluate_hf.py \
    --checkpoint "${checkpoint}" --out "${out}" \
    --benchmark custom --manifest "${cgbench_dir}/questions.jsonl"

echo "Job finished at $(date); metrics: ${out}/custom/metrics.json"
