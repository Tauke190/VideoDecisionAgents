#!/bin/bash
#SBATCH --job-name=cgbench_qa
#SBATCH --output=slurmlogs/cgbench_qa-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --qos=group2-2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
# CG-Bench QA with hierarchical sampling on 2 x 48GB GPUs (tree + MCQ, open-ended, judge, metrics)
# Full run on all 12,129 questions
# Usage: QUESTIONS=/path/to/cgbench/questions.jsonl sbatch VideoJevOpenQA/run_cgbench_hierarchical.slurm
set -euo pipefail

ROOT=${ROOT:-/home/av354855/projects/VideoDecisionAgents/VideoJev}  # repo on the HPC
export QUESTIONS=${QUESTIONS:-$ROOT/data/benchmarks/cgbench/questions.jsonl}  # videos/ next to it
export SAMPLE=0  # 0 = all questions
export NPROC=${SLURM_GPUS_ON_NODE:-2}
export OUT=${OUT:-VideoJevOpenQA/runs/cgbench_qa_all}
export TQDM_MININTERVAL=60

module load anaconda3
module load cuda
cd "$ROOT"
mkdir -p slurmlogs
echo "Job started at $(date) on $(hostname)"
nvidia-smi

bash VideoJevOpenQA/run_cgbench_hierarchical.sh

echo "metrics: $ROOT/$OUT/cgbench_qa.md"
echo "Job finished at $(date)"
