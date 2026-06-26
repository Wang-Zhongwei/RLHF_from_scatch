"""
RLHF from Scratch on DistilGPT2 — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_distilgpt2_tokenizer ──
from transformers import AutoTokenizer

def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    """
    Loads and returns the Hugging Face tokenizer for the given model name.
    """
    return AutoTokenizer.from_pretrained(model_name)

# ── Step 002  load_distilgpt2_model ──
from transformers import AutoModelForCausalLM

def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    """
    Loads a causal language model by name and returns it in eval mode.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model

# ── Step 003  set_pad_token_to_eos ──
# TODO: solve this step to fill in set_pad_token_to_eos
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0003)

# ── Step 004  generate_and_decode ──
# TODO: solve this step to fill in generate_and_decode
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0004)

# ── Step 005  greedy_decode ──
# TODO: solve this step to fill in greedy_decode
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0005)

# ── Step 006  sample_with_temperature ──
# TODO: solve this step to fill in sample_with_temperature
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0006)

# ── Step 007  top_k_filter ──
# TODO: solve this step to fill in top_k_filter
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0007)

# ── Step 008  top_p_filter ──
# TODO: solve this step to fill in top_p_filter
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0008)

# ── Step 009  build_synthetic_instruction_dataset ──
# TODO: solve this step to fill in build_synthetic_instruction_dataset
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0009)

# ── Step 010  format_example ──
# TODO: solve this step to fill in format_example
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0010)

# ── Step 011  apply_template ──
# TODO: solve this step to fill in apply_template
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0011)

# ── Step 012  tokenize_example ──
# TODO: solve this step to fill in tokenize_example
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0012)

# ── Step 013  build_labels ──
# TODO: solve this step to fill in build_labels
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0013)

# ── Step 014  mask_prompt_labels ──
# TODO: solve this step to fill in mask_prompt_labels
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0014)

# ── Step 015  pad_batch ──
# TODO: solve this step to fill in pad_batch
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0015)

# ── Step 016  make_attention_mask ──
# TODO: solve this step to fill in make_attention_mask
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0016)

# ── Step 017  collate_lm_batch ──
# TODO: solve this step to fill in collate_lm_batch
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0017)

# ── Step 018  iterate_minibatches ──
# TODO: solve this step to fill in iterate_minibatches
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0018)

# ── Step 019  train_val_split ──
# TODO: solve this step to fill in train_val_split
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0019)

# ── Step 020  shift_logits_and_labels ──
# TODO: solve this step to fill in shift_logits_and_labels
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0020)

# ── Step 021  cross_entropy_loss ──
# TODO: solve this step to fill in cross_entropy_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0021)

# ── Step 022  adamw_update ──
# TODO: solve this step to fill in adamw_update
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0022)

# ── Step 023  linear_warmup_schedule ──
# TODO: solve this step to fill in linear_warmup_schedule
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0023)

# ── Step 024  clip_grad_norm ──
# TODO: solve this step to fill in clip_grad_norm
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0024)

# ── Step 025  accumulate_gradients ──
# TODO: solve this step to fill in accumulate_gradients
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0025)

# ── Step 026  sft_train_step ──
# TODO: solve this step to fill in sft_train_step
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0026)

# ── Step 027  evaluate_loss ──
# TODO: solve this step to fill in evaluate_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0027)

# ── Step 028  lora_delta ──
# TODO: solve this step to fill in lora_delta
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0028)

# ── Step 029  lora_linear_forward ──
# TODO: solve this step to fill in lora_linear_forward
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0029)

# ── Step 030  init_lora_weights ──
# TODO: solve this step to fill in init_lora_weights
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0030)

# ── Step 031  freeze_base_params ──
# TODO: solve this step to fill in freeze_base_params
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0031)

# ── Step 032  count_trainable_params ──
# TODO: solve this step to fill in count_trainable_params
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0032)

# ── Step 033  merge_lora ──
# TODO: solve this step to fill in merge_lora
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0033)

# ── Step 034  build_synthetic_preference_dataset ──
# TODO: solve this step to fill in build_synthetic_preference_dataset
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0034)

# ── Step 035  format_preference ──
# TODO: solve this step to fill in format_preference
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0035)

# ── Step 036  reward_head_forward ──
# TODO: solve this step to fill in reward_head_forward
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0036)

# ── Step 037  pairwise_reward_loss ──
# TODO: solve this step to fill in pairwise_reward_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0037)

# ── Step 038  reward_bce_loss ──
# TODO: solve this step to fill in reward_bce_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0038)

# ── Step 039  pairwise_accuracy ──
# TODO: solve this step to fill in pairwise_accuracy
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0039)

# ── Step 040  reward_train_step ──
# TODO: solve this step to fill in reward_train_step
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0040)

# ── Step 041  sequence_logprob ──
# TODO: solve this step to fill in sequence_logprob
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0041)

# ── Step 042  per_token_kl ──
# TODO: solve this step to fill in per_token_kl
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0042)

# ── Step 043  compute_returns ──
# TODO: solve this step to fill in compute_returns
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0043)

# ── Step 044  gae_advantages ──
# TODO: solve this step to fill in gae_advantages
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0044)

# ── Step 045  policy_ratio ──
# TODO: solve this step to fill in policy_ratio
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0045)

# ── Step 046  clipped_surrogate ──
# TODO: solve this step to fill in clipped_surrogate
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0046)

# ── Step 047  value_function_loss ──
# TODO: solve this step to fill in value_function_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0047)

# ── Step 048  entropy_bonus ──
# TODO: solve this step to fill in entropy_bonus
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0048)

# ── Step 049  ppo_loss ──
# TODO: solve this step to fill in ppo_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0049)

# ── Step 050  kl_penalized_reward ──
# TODO: solve this step to fill in kl_penalized_reward
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0050)

# ── Step 051  batch_sequence_logprob ──
# TODO: solve this step to fill in batch_sequence_logprob
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0051)

# ── Step 052  dpo_logratios ──
# TODO: solve this step to fill in dpo_logratios
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0052)

# ── Step 053  dpo_ref_logratios ──
# TODO: solve this step to fill in dpo_ref_logratios
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0053)

# ── Step 054  dpo_loss ──
# TODO: solve this step to fill in dpo_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0054)

# ── Step 055  ipo_loss ──
# TODO: solve this step to fill in ipo_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0055)

# ── Step 056  kto_loss ──
# TODO: solve this step to fill in kto_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0056)

# ── Step 057  orpo_loss ──
# TODO: solve this step to fill in orpo_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0057)

# ── Step 058  simpo_loss ──
# TODO: solve this step to fill in simpo_loss
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0058)

# ── Step 059  build_eval_prompt_set ──
# TODO: solve this step to fill in build_eval_prompt_set
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0059)

# ── Step 060  generate_completions ──
# TODO: solve this step to fill in generate_completions
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0060)

# ── Step 061  score_with_reward ──
# TODO: solve this step to fill in score_with_reward
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0061)

# ── Step 062  win_rate ──
# TODO: solve this step to fill in win_rate
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0062)

# ── Step 063  stream_tokens ──
# TODO: solve this step to fill in stream_tokens
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0063)

# ── Step 064  apply_stop_tokens ──
# TODO: solve this step to fill in apply_stop_tokens
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0064)

# ── Step 065  chat ──
# TODO: solve this step to fill in chat
#       (see /projects/rlhf-from-scratch-on-distilgpt2/step/rlhf-from-scratch-on-distilgpt2-0065)

# ── Scaffold (runner) ──
"""End-to-end RLHF-from-scratch demo on a tiny GPT-2: SFT -> reward modeling -> PPO/DPO -> chat."""
import numpy as np
import torch

from solution import (
    load_distilgpt2_tokenizer,
    load_distilgpt2_model,
    set_pad_token_to_eos,
    generate_and_decode,
    greedy_decode,
    sample_with_temperature,
    top_k_filter,
    top_p_filter,
    build_synthetic_instruction_dataset,
    format_example,
    apply_template,
    tokenize_example,
    build_labels,
    mask_prompt_labels,
    pad_batch,
    make_attention_mask,
    collate_lm_batch,
    iterate_minibatches,
    train_val_split,
    shift_logits_and_labels,
    cross_entropy_loss,
    adamw_update,
    linear_warmup_schedule,
    clip_grad_norm,
    accumulate_gradients,
    sft_train_step,
    evaluate_loss,
    lora_delta,
    lora_linear_forward,
    init_lora_weights,
    freeze_base_params,
    count_trainable_params,
    merge_lora,
    build_synthetic_preference_dataset,
    format_preference,
    reward_head_forward,
    pairwise_reward_loss,
    reward_bce_loss,
    pairwise_accuracy,
    reward_train_step,
    sequence_logprob,
    per_token_kl,
    compute_returns,
    gae_advantages,
    policy_ratio,
    clipped_surrogate,
    value_function_loss,
    entropy_bonus,
    ppo_loss,
    kl_penalized_reward,
    batch_sequence_logprob,
    dpo_ref_logratios,
    dpo_loss,
    ipo_loss,
    kto_loss,
    orpo_loss,
    simpo_loss,
    build_eval_prompt_set,
    generate_completions,
    score_with_reward,
    win_rate,
    stream_tokens,
    apply_stop_tokens,
    chat,
)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # 1) Tokenizer + base model
    tokenizer = load_distilgpt2_tokenizer()
    set_pad_token_to_eos(tokenizer)
    model = load_distilgpt2_model()
    pad_id = tokenizer.pad_token_id
    print(f"Loaded tiny-gpt2; vocab={len(tokenizer)}, pad==eos? {tokenizer.pad_token == tokenizer.eos_token}")

    # 2) Baseline generation (un-aligned)
    base_out = generate_and_decode(model, tokenizer, "Hello, how are you?", max_new_tokens=8)
    print("Base completion:", repr(base_out))

    # 3) Build + tokenize SFT data
    sft_data = build_synthetic_instruction_dataset()
    train_data, val_data = train_val_split(sft_data, val_ratio=0.25, seed=0)
    train_texts = apply_template(train_data)
    val_texts = apply_template(val_data)

    def make_batches(texts, bs=2):
        examples = []
        for t in texts:
            enc = tokenize_example(tokenizer, t, max_length=32)
            ids = enc["input_ids"] if isinstance(enc, dict) else enc
            labels = build_labels(ids)
            examples.append({"input_ids": ids, "labels": labels})
        for mb in iterate_minibatches(examples, batch_size=bs, seed=0):
            yield collate_lm_batch(mb, pad_id)

    # 4) Short SFT loop and watch loss drop
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    losses = []
    for step in range(3):
        for batch in make_batches(train_texts, bs=2):
            loss = sft_train_step(model, batch, optimizer)
            losses.append(float(loss))
    val_loss = evaluate_loss(model, list(make_batches(val_texts, bs=2)))
    print(f"SFT train losses: {[round(l, 3) for l in losses[:6]]}... val_loss={float(val_loss):.3f}")

    # 5) Reward model: train a tiny head on synthetic preferences
    pref_data = build_synthetic_preference_dataset(num_examples=6, seed=0)
    hidden = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size
    reward_head = torch.nn.Linear(hidden, 1)
    rm_opt = torch.optim.AdamW(reward_head.parameters(), lr=1e-3)

    class _HiddenBackbone:
        """Adapter: the reward_train_step contract expects a callable returning a
        hidden-state tensor of shape (B, T, H), but we have a full LM head model."""
        def __init__(self, m):
            self.m = m
        def __call__(self, ids, attention_mask=None):
            out = self.m(ids, attention_mask=attention_mask, output_hidden_states=True)
            return out.hidden_states[-1]

    backbone = _HiddenBackbone(model)

    def _build_pref_batch(pref, tok, max_length=32):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        chosen_texts = [ex["prompt"] + " " + ex["chosen"] for ex in pref]
        rejected_texts = [ex["prompt"] + " " + ex["rejected"] for ex in pref]
        ce = tok(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        re_ = tok(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return {
            "chosen_input_ids": ce["input_ids"],
            "chosen_attention_mask": ce["attention_mask"],
            "rejected_input_ids": re_["input_ids"],
            "rejected_attention_mask": re_["attention_mask"],
        }

    rm_batch = _build_pref_batch(pref_data, tokenizer)
    rm_out = None
    for _ in range(2):
        rm_out = reward_train_step(backbone, reward_head, rm_batch, rm_opt)
    rm_loss = rm_out["loss"] if isinstance(rm_out, dict) else float(rm_out)
    print(f"Reward head trained; final RM loss ~ {float(rm_loss):.3f}")

    # 6) Compare aligned vs base via reward-model win-rate
    eval_prompts = build_eval_prompt_set()[:3]
    comps_aligned = generate_completions(model, tokenizer, eval_prompts, max_new_tokens=8)
    base_again = load_distilgpt2_model()
    comps_base = generate_completions(base_again, tokenizer, eval_prompts, max_new_tokens=8)

    # score_with_reward expects a dict bundle, not a bare nn.Linear.
    reward_bundle = {
        "model": model,
        "weight": reward_head.weight,
        "bias": reward_head.bias,
    }
    scored = [
        (score_with_reward(reward_bundle, tokenizer, p, c_a),
         score_with_reward(reward_bundle, tokenizer, p, c_b))
        for p, c_a, c_b in zip(eval_prompts, comps_aligned, comps_base)
    ]
    wins = sum(1 for a, b in scored if float(a) > float(b))
    print(f"Aligned beats base on {wins}/{len(scored)} prompts (reward-model judged)")

    # 7) Minimal chat interface
    reply = chat(model, tokenizer, "Say hi.", system_prompt="You are helpful.", max_new_tokens=8)
    print("Chat reply:", repr(reply))
