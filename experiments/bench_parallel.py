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
import json
import os
import time

import torch

from rlhf import load_model, load_tokenizer, set_pad_token_to_eos
from rlhf.parallel import (
    setup_distributed, cleanup_distributed, is_main_process,
    wrap_fsdp, gpt2_block_cls, peak_memory_mb,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HF model id (default: rlhf DEFAULT_MODEL). Use gpt2-medium for a "
                         "meaningful memory-sharding curve.")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--no-fsdp", action="store_true", help="plain model (baseline)")
    ap.add_argument("--out", default=None,
                    help="JSON file to record {world_size: {peak_mem_mb, throughput,...}}")
    args = ap.parse_args()

    rank, local_rank, world_size, device = setup_distributed()
    load_kwargs = {"model_name": args.model} if args.model else {}
    tokenizer = set_pad_token_to_eos(load_tokenizer(**load_kwargs))
    model = load_model(**load_kwargs).to(device)

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
        fsdp_on = not args.no_fsdp and world_size > 1
        peak_mb = peak_memory_mb()
        print(f"world_size={world_size} fsdp={fsdp_on} "
              f"peak_mem={peak_mb:.1f}MB "
              f"throughput={tok_per_s:,.0f} tok/s/rank")
        if args.out:
            # Sequential SLURM sweep (world_size 1,2,4) -> merge into one dict, keyed by
            # world_size, so plot_showcase can draw the peak-memory-vs-GPUs curve.
            record = {
                "peak_mem_mb": peak_mb, "throughput_tok_s": tok_per_s, "fsdp": fsdp_on,
                "model": args.model or "default", "batch_size": args.batch_size,
                "seq_len": args.seq_len,
            }
            data = {}
            if os.path.exists(args.out):
                with open(args.out) as f:
                    data = json.load(f)
            data[str(world_size)] = record
            with open(args.out, "w") as f:
                json.dump(data, f, indent=2)
            print("wrote", args.out)
    cleanup_distributed()


if __name__ == "__main__":
    main()
