# RLHF / GRPO from scratch

A full post-training stack built up from primitives — decoding, SFT, LoRA,
reward modeling, and the alignment objectives (**PPO, GRPO, DPO, IPO, KTO,
ORPO, SimPO**) — plus the **model-parallel internals** (Megatron-style
tensor parallelism and FSDP sharding) that let these run on models too large for
a single GPU. Everything is implemented directly in PyTorch: the AdamW update,
the GAE recursion, the clipped surrogate, the Bradley–Terry reward loss, the
group-relative advantage, and the tensor-parallel collectives are all written
out rather than imported from a training library.

Runs end-to-end on CPU with `sshleifer/tiny-gpt2` for fast iteration/CI, and
scales to real GPT-2 / Llama checkpoints on multi-GPU (torchrun + NCCL) by
changing one model name.

## Layout

```
rlhf/
  models.py          load tokenizer/model, pad-token handling
  decoding.py        greedy / temperature / top-k / top-p, streaming, chat
  data.py            synthetic instruction + preference sets, templating
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
    dist.py            torchrun/NCCL process-group bootstrap
    tensor_parallel.py Megatron column/row-parallel linears (custom autograd)
    fsdp.py            ZeRO-3 FSDP wrapping + peak-memory measurement
experiments/
  pipeline_demo.py            end-to-end SFT->RM->win-rate->chat smoke run
  exp1_alignment_frontier.py  reward-vs-KL frontier: PPO vs GRPO vs DPO
  exp2_grpo_sample_efficiency.py  GRPO vs PPO reward-per-step + peak memory
  bench_parallel.py           FSDP memory/throughput scaling under torchrun
scripts/                      slurm jobs for coe-hpc3
tests/                        hermetic unit tests for the math + TP correctness
```

## Quickstart

```bash
pip install -r requirements.txt
pytest -q                              # fast math/TP unit tests (no downloads)
python -m experiments.pipeline_demo    # end-to-end pipeline on tiny-gpt2
```

## Headline experiments

**1. Alignment frontier (reward vs KL).** Align one SFT checkpoint with PPO,
GRPO, and DPO across a sweep of KL/β settings, then plot mean reward-model score
against KL drift from the reference. Methods that sit up-and-to-the-left buy
more reward per unit of distribution shift.

```bash
python -m experiments.exp1_alignment_frontier --steps 200 --group-size 8
python -m experiments.plot_frontier results/frontier.json
```

**2. GRPO vs PPO sample efficiency.** Same reward model, same start; log reward
per step and peak memory. GRPO drops PPO's value network, so it carries no
critic parameters or critic optimizer state — visible as lower peak memory for
comparable reward.

```bash
python -m experiments.exp2_grpo_sample_efficiency --steps 200 --group-size 8
```

## Model-parallel benchmark (multi-GPU)

`experiments/bench_parallel.py` initializes NCCL, wraps the policy in FSDP
(params/grads/optimizer state sharded), and reports peak per-GPU memory and
throughput. Sweep the world size to see memory fall ~1/N:

```bash
torchrun --standalone --nproc_per_node=1 -m experiments.bench_parallel   # baseline
torchrun --standalone --nproc_per_node=4 -m experiments.bench_parallel   # 4-way FSDP
```

The tensor-parallel linears in `rlhf/parallel/tensor_parallel.py` shard a single
matmul's weight across ranks with custom autograd collectives (all-gather on the
column-parallel output, all-reduce on the row-parallel output), verified against
a dense `nn.Linear` in `tests/test_tensor_parallel.py`.

## On coe-hpc3

```bash
git clone <repo> && cd RLHF_from_scatch
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
sbatch scripts/train_hpc.slurm       # the two headline experiments (1 GPU)
sbatch scripts/bench_parallel.slurm  # FSDP scaling sweep (4 GPUs)
```
