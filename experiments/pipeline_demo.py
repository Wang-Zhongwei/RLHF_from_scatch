"""End-to-end smoke demo: SFT -> reward modeling -> reward-judged win-rate -> chat.

Runs in seconds on CPU with tiny-gpt2. This is the "does the whole pipeline wire
together" check; the headline results live in exp1/exp2.
"""
import numpy as np
import torch

from rlhf import (
    load_tokenizer, load_model, set_pad_token_to_eos,
    generate_and_decode, build_synthetic_instruction_dataset, apply_template,
    tokenize_example, build_labels, collate_lm_batch, iterate_minibatches,
    train_val_split, sft_train_step, evaluate_loss,
    build_synthetic_preference_dataset, reward_train_step,
    build_eval_prompt_set, generate_completions, score_with_reward, chat,
)


class HiddenBackbone:
    """Adapter: reward_train_step wants a callable returning (B, T, H) hidden
    states, but we have a full LM-head model."""

    def __init__(self, m):
        self.m = m

    def __call__(self, input_ids, attention_mask=None):
        out = self.m(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-1]


def build_lm_batches(texts, tokenizer, pad_id, bs=2):
    examples = []
    for t in texts:
        enc = tokenize_example(tokenizer, t, max_length=32)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc
        examples.append({"input_ids": ids, "labels": build_labels(ids)})
    for mb in iterate_minibatches(examples, batch_size=bs, seed=0):
        yield collate_lm_batch(mb, pad_id)


def build_pref_batch(pref, tokenizer, max_length=32):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    chosen = [ex["prompt"] + " " + ex["chosen"] for ex in pref]
    rejected = [ex["prompt"] + " " + ex["rejected"] for ex in pref]
    ce = tokenizer(chosen, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    re_ = tokenizer(rejected, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {
        "chosen_input_ids": ce["input_ids"],
        "chosen_attention_mask": ce["attention_mask"],
        "rejected_input_ids": re_["input_ids"],
        "rejected_attention_mask": re_["attention_mask"],
    }


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    tokenizer = load_tokenizer()
    set_pad_token_to_eos(tokenizer)
    model = load_model()
    pad_id = tokenizer.pad_token_id
    print(f"Loaded tiny-gpt2; vocab={len(tokenizer)}, pad==eos? "
          f"{tokenizer.pad_token == tokenizer.eos_token}")

    print("Base completion:", repr(generate_and_decode(model, tokenizer, "Hello, how are you?", 8)))

    sft_data = build_synthetic_instruction_dataset()
    train_data, val_data = train_val_split(sft_data, val_ratio=0.25, seed=0)
    train_texts, val_texts = apply_template(train_data), apply_template(val_data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    losses = []
    for _ in range(3):
        for batch in build_lm_batches(train_texts, tokenizer, pad_id, bs=2):
            losses.append(float(sft_train_step(model, batch, optimizer)))
    val_loss = evaluate_loss(model, list(build_lm_batches(val_texts, tokenizer, pad_id, bs=2)))
    print(f"SFT train losses: {[round(l, 3) for l in losses[:6]]}... val_loss={val_loss:.3f}")

    pref_data = build_synthetic_preference_dataset(num_examples=6, seed=0)
    hidden = getattr(model.config, "n_embd", None) or model.config.hidden_size
    reward_head = torch.nn.Linear(hidden, 1)
    rm_opt = torch.optim.AdamW(reward_head.parameters(), lr=1e-3)
    backbone = HiddenBackbone(model)
    rm_batch = build_pref_batch(pref_data, tokenizer)
    rm_out = None
    for _ in range(2):
        rm_out = reward_train_step(backbone, reward_head, rm_batch, rm_opt)
    print(f"Reward head trained; final RM loss ~ {rm_out['loss']:.3f} acc={rm_out['accuracy']:.2f}")

    eval_prompts = build_eval_prompt_set()[:3]
    comps_aligned = generate_completions(model, tokenizer, eval_prompts, max_new_tokens=8)
    comps_base = generate_completions(load_model(), tokenizer, eval_prompts, max_new_tokens=8)
    reward_bundle = {"model": model, "weight": reward_head.weight, "bias": reward_head.bias}
    scored = [
        (score_with_reward(reward_bundle, tokenizer, p, a),
         score_with_reward(reward_bundle, tokenizer, p, b))
        for p, a, b in zip(eval_prompts, comps_aligned, comps_base)
    ]
    wins = sum(1 for a, b in scored if a > b)
    print(f"Aligned beats base on {wins}/{len(scored)} prompts (reward-model judged)")

    reply = chat(model, tokenizer, "Say hi.", system_prompt="You are helpful.", max_new_tokens=8)
    print("Chat reply:", repr(reply))


if __name__ == "__main__":
    main()
