"""Model + tokenizer loading and pad-token handling.

Kept intentionally backbone-agnostic: everything downstream (SFT, reward,
PPO/GRPO, DPO-family) only assumes a HuggingFace causal-LM + tokenizer, so the
same pipeline scales from ``sshleifer/tiny-gpt2`` (CI) to real GPT-2 / Llama
checkpoints on GPU by changing ``model_name``.
"""
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "sshleifer/tiny-gpt2"


def _resolve(model_name: str) -> str:
    """Resolve a HF repo id to its local snapshot dir when it's already cached.

    transformers 4.57 still issues a ``model_info`` network call inside
    ``from_pretrained`` even under ``HF_HUB_OFFLINE`` (and even with
    ``local_files_only=True``), which errors on air-gapped compute nodes. Passing an
    explicit local directory sidesteps the call. Local paths and not-yet-cached repo
    ids (e.g. first download on the login node) pass through unchanged.
    """
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    """Load the HuggingFace tokenizer for ``model_name``."""
    return AutoTokenizer.from_pretrained(_resolve(model_name))


def load_model(model_name: str = DEFAULT_MODEL):
    """Load a causal LM and return it in eval mode."""
    model = AutoModelForCausalLM.from_pretrained(_resolve(model_name))
    model.eval()
    return model


def set_pad_token_to_eos(tokenizer):
    """GPT-2-family tokenizers ship without a pad token; reuse EOS for padding."""
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Backwards-compatible aliases (original scaffold names).
load_distilgpt2_tokenizer = load_tokenizer
load_distilgpt2_model = load_model
