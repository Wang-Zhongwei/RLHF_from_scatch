"""RLHF-from-scratch: SFT -> reward modeling -> PPO / GRPO / DPO-family, with
from-scratch tensor-parallel + FSDP model-parallel internals.

The package is organized by pipeline stage; this module re-exports the full flat
API so notebooks and experiments can ``from rlhf import <fn>`` without caring
which file a function lives in.
"""
from .models import (
    load_tokenizer, load_model, set_pad_token_to_eos,
    load_distilgpt2_tokenizer, load_distilgpt2_model,
)
from .decoding import (
    generate_and_decode, greedy_decode, sample_with_temperature,
    top_k_filter, top_p_filter, stream_tokens, apply_stop_tokens, chat,
)
from .data import (
    build_synthetic_instruction_dataset, format_example, apply_template,
    build_synthetic_preference_dataset, format_preference, build_eval_prompt_set,
    load_preference_dataset, load_prompt_set,
)
from .tokenization import (
    tokenize_example, build_labels, mask_prompt_labels, pad_batch,
    make_attention_mask, collate_lm_batch, iterate_minibatches, train_val_split,
)
from .losses import shift_logits_and_labels, cross_entropy_loss
from .optim import (
    adamw_update, linear_warmup_schedule, clip_grad_norm, accumulate_gradients,
)
from .sft import sft_train_step, evaluate_loss
from .lora import (
    lora_delta, lora_linear_forward, init_lora_weights, freeze_base_params,
    count_trainable_params, merge_lora,
)
from .reward import (
    reward_head_forward, pairwise_reward_loss, reward_bce_loss,
    pairwise_accuracy, reward_train_step, save_reward_model, load_reward_model,
)
from .ppo import (
    sequence_logprob, per_token_kl, compute_returns, gae_advantages,
    policy_ratio, clipped_surrogate, value_function_loss, entropy_bonus,
    ppo_loss, kl_penalized_reward,
)
from .grpo import group_relative_advantages, k3_kl, grpo_loss
from .preference import (
    batch_sequence_logprob, dpo_logratios, dpo_ref_logratios, dpo_loss,
    ipo_loss, kto_loss, orpo_loss, simpo_loss,
)
from .eval import generate_completions, score_with_reward, win_rate

__all__ = [name for name in globals() if not name.startswith("_")]
