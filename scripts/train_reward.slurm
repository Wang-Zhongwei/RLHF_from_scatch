#!/bin/bash
#SBATCH --job-name=rlhf-rm
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/rm_%j.out

# Train a real reward model (backbone + scalar head) and save it to disk.
# Override defaults via sbatch --export, e.g. RM_MODEL=gpt2 RM_MAX=3000.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

source /etc/profile.d/lmod.sh
module load python3/3.12.12 ml/torch/2.6

# Compute nodes are air-gapped; models + dataset were pre-cached to the shared HF cache.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p logs results

nvidia-smi || true

# Set RM_FREEZE=1 to train only the linear head over a frozen backbone (linear probe).
python -m experiments.train_reward_model \
  --model "${RM_MODEL:-gpt2-medium}" --dataset "${RM_DATASET:-Dahoas/rm-static}" \
  --max-examples "${RM_MAX:-10000}" --epochs "${RM_EPOCHS:-2}" \
  --batch-size "${RM_BS:-8}" --max-length "${RM_MAXLEN:-512}" \
  --lr "${RM_LR:-1e-5}" ${RM_FREEZE:+--freeze-backbone} --out "${RM_OUT:-results/reward_model}"
