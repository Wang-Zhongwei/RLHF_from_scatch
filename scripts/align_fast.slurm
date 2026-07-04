#!/bin/bash
#SBATCH --job-name=rlhf-fast
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/fast_%j.out

# Fast validation tier (Stage-1 gate): train a small RM on gpt2, then run a reduced
# alignment sweep. Confirms real RM val-accuracy + non-zero KL separation before we
# commit to the full gpt2-medium run. Writes to results_fast/ to avoid clobbering.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

source /etc/profile.d/lmod.sh
module load python3/3.12.12 ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p logs results_fast

nvidia-smi || true

RM=results_fast/reward_model
# Set RM_FREEZE=1 to train only the linear head over a frozen backbone (linear probe).
python -m experiments.train_reward_model \
  --model gpt2 --dataset Dahoas/rm-static \
  --max-examples 5000 --epochs 2 --batch-size 16 --max-length 384 \
  ${RM_FREEZE:+--freeze-backbone} --out "$RM"

python -m experiments.exp1_alignment_frontier \
  --rm-dir "$RM" --dataset Dahoas/rm-static \
  --steps 100 --group-size 8 --max-new-tokens 32 \
  --n-train-prompts 32 --n-eval-prompts 16 \
  --value-warmup 10 --vf-lr 1e-3 \
  --kl-coefs 0.0 0.05 0.1 --dpo-betas 0.05 0.1 0.5 \
  --out results_fast/frontier.json

python -m experiments.exp2_grpo_sample_efficiency \
  --rm-dir "$RM" --dataset Dahoas/rm-static \
  --steps 100 --group-size 8 --max-new-tokens 32 \
  --n-train-prompts 32 --n-eval-prompts 16 \
  --value-warmup 10 --vf-lr 1e-3 \
  --out results_fast/sample_eff.json

python -m experiments.plot_frontier results_fast/frontier.json results_fast/sample_eff.json
