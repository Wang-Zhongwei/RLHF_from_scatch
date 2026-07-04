"""Train a real reward model (backbone + scalar head) on a preference dataset.

Standard reward modeling: fine-tune a causal-LM backbone end-to-end with a single
linear head that maps the last-token hidden state to a scalar, under the
Bradley-Terry pairwise loss (rlhf.pairwise_reward_loss). The trained RM is saved
to disk (rlhf.save_reward_model) so the alignment experiments load a *meaningful*
reward signal instead of the synthetic toy head built in-process by build_harness.

    python -m experiments.train_reward_model \
        --model gpt2-medium --dataset Dahoas/rm-static \
        --max-examples 10000 --epochs 2 --batch-size 16 --out results/reward_model

Runs on GPU when available. Reports held-out pairwise accuracy each epoch — the
signal that tells you the RM actually learned (random = 0.5).
"""
import argparse
import json
import os

import torch

from rlhf import (
    load_tokenizer, load_model, set_pad_token_to_eos,
    load_preference_dataset, reward_head_forward, pairwise_reward_loss,
    pairwise_accuracy, clip_grad_norm, linear_warmup_schedule, save_reward_model,
)
from rlhf.tokenization import iterate_minibatches, train_val_split
from experiments.pipeline_demo import HiddenBackbone, build_pref_batch


def _scores(backbone, head, batch, pad_id):
    """Bradley-Terry chosen/rejected scalar rewards for one tokenized batch."""
    ch = backbone(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"])
    rj = backbone(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])
    ci = (batch["chosen_input_ids"] != pad_id).sum(dim=-1) - 1
    ri = (batch["rejected_input_ids"] != pad_id).sum(dim=-1) - 1
    ch_last = ch[torch.arange(ch.shape[0]), ci]
    rj_last = rj[torch.arange(rj.shape[0]), ri]
    return (reward_head_forward(ch_last, head.weight, head.bias),
            reward_head_forward(rj_last, head.weight, head.bias))


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(backbone, head, examples, tokenizer, device, pad_id, batch_size, max_length):
    accs, losses, n = [], [], 0
    for mb in iterate_minibatches(examples, batch_size, seed=0):
        batch = _to_device(build_pref_batch(mb, tokenizer, max_length), device)
        cr, rr = _scores(backbone, head, batch, pad_id)
        accs.append(pairwise_accuracy(cr, rr).item() * len(mb))
        losses.append(pairwise_reward_loss(cr, rr).item() * len(mb))
        n += len(mb)
    return sum(losses) / n, sum(accs) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dataset", default="Dahoas/rm-static")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the linear head over frozen features (linear probe)")
    ap.add_argument("--out", default="results/reward_model")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} dataset={args.dataset}")

    tokenizer = set_pad_token_to_eos(load_tokenizer(args.model))
    # Dialogue prompts are long; keep the *response* end (which the RM scores) by
    # truncating from the left, otherwise right-truncation drops the response and
    # chosen/rejected collapse to the same prompt prefix.
    tokenizer.truncation_side = "left"
    pad_id = tokenizer.pad_token_id

    backbone_model = load_model(args.model).to(device)
    if args.freeze_backbone:
        for p in backbone_model.parameters():
            p.requires_grad_(False)
        backbone_model.eval()      # linear probe: no dropout, backbone never updates
    else:
        backbone_model.train()
    backbone = HiddenBackbone(backbone_model)
    hidden = getattr(backbone_model.config, "n_embd", None) or backbone_model.config.hidden_size
    head = torch.nn.Linear(hidden, 1).to(device)

    data = load_preference_dataset(args.dataset, "train", args.max_examples, args.seed)
    train, val = train_val_split(data, val_ratio=args.val_ratio, seed=args.seed)
    print(f"pairs: train={len(train)} val={len(val)} "
          f"(pad_id={pad_id}, hidden={hidden}, freeze_backbone={args.freeze_backbone})")

    trainable = list(head.parameters()) if args.freeze_backbone \
        else list(backbone_model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    steps_per_epoch = max(1, (len(train) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    step = 0
    best_val = 0.0
    for epoch in range(args.epochs):
        for mb in iterate_minibatches(train, args.batch_size, seed=args.seed + epoch):
            batch = _to_device(build_pref_batch(mb, tokenizer, args.max_length), device)
            cr, rr = _scores(backbone, head, batch, pad_id)
            loss = pairwise_reward_loss(cr, rr) 
            opt.zero_grad()
            loss.backward()
            clip_grad_norm([p.grad for p in trainable if p.grad is not None], args.max_grad_norm)
            for g in opt.param_groups:
                g["lr"] = args.lr * linear_warmup_schedule(step, warmup_steps)
            opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                acc = pairwise_accuracy(cr, rr).item()
                print(f"  step {step}/{total_steps} loss={loss.item():.4f} train_acc={acc:.3f}", flush=True)
        backbone_model.eval()
        val_loss, val_acc = evaluate(backbone, head, val, tokenizer, device, pad_id,
                                     args.batch_size, args.max_length)
        if not args.freeze_backbone:
            backbone_model.train()
        best_val = max(best_val, val_acc)
        print(f"[epoch {epoch+1}] val_loss={val_loss:.4f} val_pairwise_acc={val_acc:.3f}", flush=True)

    meta = {"model": args.model, "dataset": args.dataset, "hidden": hidden,
            "max_length": args.max_length, "train_pairs": len(train), "val_pairs": len(val),
            "val_pairwise_acc": best_val}
    save_reward_model(args.out, backbone_model, head, meta)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.out} (val_pairwise_acc={best_val:.3f})")


if __name__ == "__main__":
    main()
