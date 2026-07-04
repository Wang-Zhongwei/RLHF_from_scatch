#!/bin/bash
#SBATCH --job-name=rlhf-align
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/align_%j.out

# Run the two headline experiments on coe-hpc3 and render the figures.
set -euo pipefail
module load cuda 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"
mkdir -p logs results

# Experiment 1: reward-vs-KL alignment frontier (PPO / GRPO / DPO).
python -m experiments.exp1_alignment_frontier \
  --steps 200 --group-size 8 --max-new-tokens 32 \
  --betas 0.01 0.05 0.1 0.5 --out results/frontier.json

# Experiment 2: GRPO vs PPO sample efficiency.
python -m experiments.exp2_grpo_sample_efficiency \
  --steps 200 --group-size 8 --max-new-tokens 32 --out results/sample_eff.json

python -m experiments.plot_frontier results/frontier.json results/sample_eff.json
