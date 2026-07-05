"""From-scratch GRPO on GSM8K with a verifiable reward, benchmarked across
single-GPU / DDP / FSDP.

  torchrun --standalone --nproc_per_node=4 -m experiments.gsm8k_grpo \
      --strategy fsdp --model Qwen/Qwen2.5-0.5B-Instruct --steps 300

Reward is a *function*, not a learned model: sample G completions per question, score
each by whether its final number matches the gold answer (``rlhf.gsm8k``), normalize the
rewards within the group (GRPO's critic-free baseline), and update the policy on the
recomputed log-probs of the tokens it generated. No TRL/verl — the GRPO math is
``rlhf/grpo.py``, the rollout is ``rlhf/sampling.py``.

The ``--strategy`` switch wraps the *same* training loop in nothing / DDP / FSDP so we can
report peak memory-per-GPU and throughput for each and show the DDP-vs-FSDP tradeoff.
"""
import argparse
import contextlib
import json
import os
import time

import torch

from rlhf import load_model, load_tokenizer, set_pad_token_to_eos
from rlhf import group_relative_advantages, grpo_loss
from rlhf.gsm8k import load_gsm8k, format_prompt, extract_pred, is_correct
from rlhf.sampling import sample_completions, gen_logprobs
from rlhf.parallel import (
    setup_distributed, cleanup_distributed, is_main_process, all_reduce_mean,
    wrap_ddp, wrap_fsdp, gpt2_block_cls, qwen_block_cls, peak_memory_mb,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _block_cls_for(model_name):
    return qwen_block_cls() if "qwen" in (model_name or "").lower() else gpt2_block_cls()


def _wrap(model, strategy, model_name, local_rank):
    if strategy == "ddp":
        return wrap_ddp(model, local_rank)
    if strategy == "fsdp":
        return wrap_fsdp(model, transformer_layer_cls=_block_cls_for(model_name))
    return model  # single


def _reward_for(seq, mask, tokenizer, gold):
    """Verifiable reward: 1.0 if the completion's final number matches gold, else 0.0."""
    gen_ids = seq[mask]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return 1.0 if is_correct(extract_pred(text), gold) else 0.0


@torch.no_grad()
def evaluate(model, tokenizer, eval_data, device, max_new_tokens, amp_ctx,
             sync_len=False, eval_temp=0.7):
    """Held-out accuracy on this rank's shard, then averaged across ranks."""
    correct = 0
    for ex in eval_data:
        prompt = format_prompt(ex["question"], tokenizer)
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        with amp_ctx():
            seqs, masks = sample_completions(
                model, ids, n=1, max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, temperature=eval_temp, top_p=0.95,
                device=device, sync_len=sync_len)
        correct += _reward_for(seqs[0], masks[0], tokenizer, ex["gold"])
    total = max(len(eval_data), 1)
    return all_reduce_mean(correct / total, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out", default="results/gsm8k_grpo.json")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny-gpt2 + tiny sizes to validate plumbing (CPU/1-GPU)")
    args = ap.parse_args()

    if args.smoke:
        args.model = "sshleifer/tiny-gpt2"
        args.steps, args.group_size, args.max_new_tokens = 3, 2, 8
        args.n_train, args.n_eval, args.eval_every = 8, 4, 2

    rank, local_rank, world_size, device = setup_distributed()
    tokenizer = set_pad_token_to_eos(load_tokenizer(args.model))
    model = load_model(args.model).to(device)
    model.train()
    model = _wrap(model, args.strategy, args.model, local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # bf16 compute for single/ddp via autocast; FSDP handles its own MixedPrecision.
    use_amp = torch.cuda.is_available() and args.strategy != "fsdp"
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp \
        else contextlib.nullcontext
    # Under FSDP every forward is a collective, so all ranks must run identical forward
    # counts: fixed generation length + equal-sized eval shards (see rlhf/sampling.py).
    fsdp_sync = args.strategy == "fsdp"

    train_data = load_gsm8k("train", n=args.n_train, seed=0)[rank::world_size]
    full_eval = load_gsm8k("test", n=args.n_eval, seed=1)
    per = max(len(full_eval) // world_size, 1)
    eval_data = full_eval[rank * per:(rank + 1) * per]
    pad_id = tokenizer.pad_token_id
    if is_main_process():
        print(f"[cfg] strategy={args.strategy} model={args.model} "
              f"world={world_size} train/rank={len(train_data)} steps={args.steps} "
              f"G={args.group_size} mnt={args.max_new_tokens}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    reward_trace, acc_trace = [], []
    t0 = None
    for step in range(args.steps):
        if step == 1:  # skip step 0 (warmup/compile) from the throughput timer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        ex = train_data[step % len(train_data)]
        prompt = format_prompt(ex["question"], tokenizer)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]

        with amp_ctx():
            seqs, masks = sample_completions(
                model, prompt_ids, n=args.group_size, max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, temperature=args.temperature,
                top_p=args.top_p, device=device, sync_len=fsdp_sync)
        rewards = [_reward_for(s, m, tokenizer, ex["gold"]) for s, m in zip(seqs, masks)]
        adv = group_relative_advantages(rewards).to(device)

        with amp_ctx():
            new_lp = gen_logprobs(model, seqs, masks, pad_id, device)
        loss = grpo_loss(new_lp, new_lp.detach(), adv, kl_coef=args.kl_coef)["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()

        reward_trace.append(all_reduce_mean(sum(rewards) / len(rewards), device))
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            acc = evaluate(model, tokenizer, eval_data, device, args.max_new_tokens, amp_ctx,
                           sync_len=fsdp_sync)
            acc_trace.append({"step": step + 1, "acc": acc})
            if is_main_process():
                print(f"[step {step + 1}/{args.steps}] train_reward="
                      f"{reward_trace[-1]:.3f} eval_acc={acc:.3f}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) if t0 else float("nan")
    timed_steps = max(args.steps - 1, 1)
    throughput = (timed_steps * args.group_size) / elapsed if elapsed == elapsed else float("nan")
    peak = peak_memory_mb()

    if is_main_process():
        rec = {
            "strategy": args.strategy, "model": args.model,
            "world_size": world_size, "group_size": args.group_size,
            "peak_mem_mb": peak, "throughput_completions_s": throughput,
            "reward_trace": reward_trace, "acc_trace": acc_trace,
            "final_acc": acc_trace[-1]["acc"] if acc_trace else None,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        data = {}
        if os.path.exists(args.out):
            with open(args.out) as f:
                data = json.load(f)
        data[args.strategy] = rec
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[done] {args.strategy}: peak_mem={peak:.0f}MB "
              f"throughput={throughput:.1f} completions/s "
              f"final_acc={rec['final_acc']}  -> {args.out}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
