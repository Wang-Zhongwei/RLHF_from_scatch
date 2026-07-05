#!/bin/bash
#SBATCH --job-name=rlhf-align
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/align_%j.out

# Standard tier: the reward-vs-KL frontier + sample-efficiency, loading the trained
# reward model at results/reward_model (produced by scripts/train_reward.slurm).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# PyTorch 2.6 (+cu126), transformers, datasets, numpy, matplotlib from the cluster module.
source /etc/profile.d/lmod.sh
module load python3/3.12.12 ml/torch/2.6

# Compute nodes are air-gapped; models + dataset were pre-cached to the shared HF cache.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p logs results

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

test -d results/reward_model || { echo "ERROR: results/reward_model missing; run scripts/train_reward.slurm first"; exit 1; }

# Experiment 1: reward-vs-KL alignment frontier (PPO / GRPO / DPO).
# Decoupled sweeps: kl_coef (additive KL penalty) for PPO/GRPO, beta (temperature) for DPO.
# Gentle LR (5e-6) + a wide kl_coef range trace the reward-vs-KL tradeoff without the
# policy reward-hacking into collapse (which pins KL at extreme values).
# Training rolls out over 128 train prompts (each swept before eval); reward/KL are scored
# on a disjoint 32-prompt held-out (test) split. PPO warms its value head for 300 steps first (critic reaches positive explained variance).
python -m experiments.exp1_alignment_frontier \
  --rm-dir results/reward_model --dataset Dahoas/rm-static \
  --steps 200 --group-size 8 --max-new-tokens 32 \
  --n-train-prompts 128 --n-eval-prompts 32 --rl-lr 5e-6 \
  --value-warmup 300 --vf-lr 1e-3 \
  --kl-coefs 0.0 0.1 0.5 2.0 --dpo-betas 0.05 0.1 0.3 0.5 \
  --out results/frontier.json

# Experiment 2: GRPO vs PPO sample efficiency (same PPO, incl. value-head warmup).
python -m experiments.exp2_grpo_sample_efficiency \
  --rm-dir results/reward_model --dataset Dahoas/rm-static \
  --steps 200 --group-size 8 --max-new-tokens 32 \
  --n-train-prompts 128 --n-eval-prompts 32 --rl-lr 5e-6 \
  --value-warmup 300 --vf-lr 1e-3 \
  --out results/sample_eff.json

python -m experiments.plot_frontier results/frontier.json results/sample_eff.json
