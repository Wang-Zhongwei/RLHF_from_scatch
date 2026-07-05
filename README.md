# RLHF / GRPO from scratch

A full post-training stack built up from primitives — decoding, SFT, LoRA,
reward modeling, and the alignment objectives (**PPO, GRPO, DPO, IPO, KTO,
ORPO, SimPO**) — plus the **data-parallel internals** (DDP, FSDP/ZeRO-3, and
Megatron-style tensor parallelism) that scale training across GPUs. Everything is
implemented directly in PyTorch: the AdamW update, the GAE recursion, the clipped
surrogate, the Bradley–Terry reward loss, the group-relative advantage, and the
distributed rollouts are all written out rather than imported from a training
library (no TRL / verl).

It runs end-to-end on CPU with `sshleifer/tiny-gpt2` for fast iteration/CI, and scales
to real checkpoints (gpt2-medium, Qwen2.5) on multi-GPU (torchrun + NCCL).

## Results

Two experiment tracks (full reproduction in [CLAUDE.md](CLAUDE.md)).

### Part 1 — Alignment on a learned reward model

![Alignment: reward vs KL knobs and GRPO vs PPO over training](figures/alignment.png)

A gpt2-medium Bradley–Terry reward model (**0.68** held-out pairwise accuracy on
`Dahoas/rm-static`, 0.5 = random) aligns one SFT policy with PPO, GRPO, and DPO:

- **A · KL penalty vs reward** — for PPO/GRPO a larger KL-penalty coefficient pulls reward
  back toward the base model; GRPO stays well above PPO throughout.
- **B · DPO runs the opposite way** — a larger DPO β pushes harder on the preference
  objective, so reward *climbs* with β (the mirror of the KL knob).
- **C · Sample efficiency** — GRPO's reward climbs above PPO's per step; PPO's learned value
  function is a noisy return estimate (negative explained variance early) and needs tuning,
  which GRPO's critic-free group baseline avoids.

### Part 2 — GRPO on GSM8K with a verifiable reward

![GRPO on GSM8K: accuracy up, DDP vs FSDP memory](figures/gsm8k_grpo.png)

The DeepSeek-R1 recipe, from scratch: GRPO with a **verifiable reward** (answer correctness —
no reward model) fine-tunes **Qwen2.5-0.5B-Instruct** on GSM8K, benchmarked across strategies.

- **Held-out accuracy 32% → 46%** optimizing a rule-based reward — real task improvement.
- **DDP vs FSDP** — FSDP (ZeRO-3) shards optimizer state to cut peak memory/GPU to **11.2 GB**
  vs DDP's **19.3 GB**; DDP keeps the throughput lead. The same training loop, three wrappers.

## Layout

```
rlhf/
  models.py          load tokenizer/model (offline-snapshot resolution), pad-token handling
  decoding.py        greedy / temperature / top-k / top-p, streaming, chat
  sampling.py        FSDP-safe batched rollout + masked log-prob recompute (GRPO)
  data.py            preference / prompt loaders (+ synthetic sets), templating
  gsm8k.py           GSM8K loader + verifiable answer-match reward
  tokenization.py    labels, prompt masking, padding, collation, splits
  losses.py          shift + masked cross-entropy
  optim.py           from-scratch AdamW, warmup, grad-clip, accumulation
  sft.py             supervised fine-tuning step + eval
  lora.py            low-rank adapters, freeze/merge
  reward.py          scalar reward head + Bradley-Terry pairwise training
  ppo.py             logprobs, KL, returns, GAE, clipped surrogate, value/entropy
  grpo.py            group-relative advantages + GRPO loss (critic-free)
  preference.py      DPO / IPO / KTO / ORPO / SimPO
  eval.py            completion generation, reward scoring, win-rate
  parallel/
    dist.py            torchrun/NCCL process-group bootstrap + metric all-reduce
    ddp.py             DistributedDataParallel wrapping (replicated)
    fsdp.py            ZeRO-3 FSDP wrapping + peak-memory measurement
    tensor_parallel.py Megatron column/row-parallel linears (custom autograd)
experiments/
  train_reward_model.py           train the scalar RM on real preferences (Bradley-Terry)
  exp1_alignment_frontier.py      reward-vs-KL frontier: PPO vs GRPO vs DPO
  exp2_grpo_sample_efficiency.py  GRPO vs PPO reward-per-step
  gsm8k_grpo.py                   GRPO on GSM8K, single / DDP / FSDP
  plot_alignment.py               render figures/alignment.png (Part 1)
  plot_gsm8k.py                   render figures/gsm8k_grpo.png (Part 2)
  pipeline_demo.py                end-to-end SFT->RM->win-rate->chat smoke run
  common.py                       shared alignment harness (build_harness, value-head PPO)
scripts/                          predownload.py + slurm jobs for coe-hpc3
figures/                          committed result figures for the README
tests/                            hermetic unit tests for the math + TP correctness
```

## Quickstart

```bash
pip install -r requirements.txt
pytest -q                              # fast math/TP unit tests (no downloads)
python -m experiments.pipeline_demo    # end-to-end pipeline on tiny-gpt2
```

## Reproduce on coe-hpc3

```bash
python scripts/predownload.py        # once, on the login node (caches models + datasets)

# Part 1 — alignment
sbatch scripts/train_reward.slurm    # reward model -> results/reward_model/
sbatch scripts/align.slurm           # frontier + sample efficiency -> figures/alignment.png

# Part 2 — GRPO on GSM8K
sbatch scripts/gsm8k_grpo.slurm      # single/DDP/FSDP sweep -> figures/gsm8k_grpo.png
```

The tensor-parallel linears in `rlhf/parallel/tensor_parallel.py` shard a single matmul's
weight across ranks with custom autograd collectives (all-gather on the column-parallel
output, all-reduce on the row-parallel output), verified against a dense `nn.Linear` in
`tests/test_tensor_parallel.py`.
