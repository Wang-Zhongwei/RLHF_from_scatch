"""Benchmark the model-parallel internals under torchrun.

Launched across N GPUs, this:
  1. initializes the NCCL process group,
  2. wraps a GPT-2 policy in FSDP (params/grads/opt-state sharded ZeRO-3 style),
  3. runs a few forward/backward steps,
  4. reports peak per-GPU memory + throughput per rank.

Run:
    torchrun --nproc_per_node=1 -m experiments.bench_parallel      # single GPU baseline
    torchrun --nproc_per_node=4 -m experiments.bench_parallel      # 4-way FSDP

Compare peak_mem across world sizes to show the ~1/N memory scaling that lets a
policy too big for one GPU fit across many. A CPU/1-process run still executes
(gloo backend) so the script is testable off-GPU.
"""
import argparse
import time

import torch

from rlhf import load_model, load_tokenizer, set_pad_token_to_eos
from rlhf.parallel import (
    setup_distributed, cleanup_distributed, is_main_process,
    wrap_fsdp, gpt2_block_cls, peak_memory_mb,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--no-fsdp", action="store_true", help="plain model (baseline)")
    args = ap.parse_args()

    rank, local_rank, world_size, device = setup_distributed()
    tokenizer = set_pad_token_to_eos(load_tokenizer())
    model = load_model().to(device)

    if not args.no_fsdp and torch.cuda.is_available() and world_size > 1:
        model = wrap_fsdp(model, transformer_layer_cls=gpt2_block_cls())
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    vocab = len(tokenizer)
    ids = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = None
    for step in range(args.steps):
        if step == 2:  # skip warmup steps in the timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) if start else float("nan")
    timed_steps = max(args.steps - 2, 1)
    tok_per_s = (timed_steps * args.batch_size * args.seq_len) / elapsed if elapsed == elapsed else float("nan")

    if is_main_process():
        print(f"world_size={world_size} fsdp={not args.no_fsdp and world_size > 1} "
              f"peak_mem={peak_memory_mb():.1f}MB "
              f"throughput={tok_per_s:,.0f} tok/s/rank")
    cleanup_distributed()


if __name__ == "__main__":
    main()
