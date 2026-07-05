"""Reward modeling: scalar head + Bradley-Terry pairwise training."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def reward_head_forward(hidden_state, weight, bias):
    """Map a final hidden state to a scalar reward via a linear projection."""
    w = weight.view(-1)  # accept both (D,) and (1, D)
    return hidden_state @ w + bias


def pairwise_reward_loss(chosen_reward, rejected_reward):
    """Bradley-Terry pairwise loss: mean(-log_sigmoid(chosen - rejected))."""
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def reward_bce_loss(chosen_reward, rejected_reward):
    """Symmetric BCE form of the pairwise reward loss (numpy reference)."""
    chosen_loss = np.logaddexp(0, -chosen_reward)
    rejected_loss = np.logaddexp(0, rejected_reward)
    return float(np.mean(0.5 * (chosen_loss + rejected_loss)))


def pairwise_accuracy(chosen_reward, rejected_reward):
    """Fraction of pairs where chosen strictly beats rejected."""
    return (chosen_reward > rejected_reward).mean(dtype=float)


def reward_train_step(model, reward_head, batch, optimizer, pad_id=0):
    """Forward chosen+rejected, score the last real token, step the optimizer.

    ``model`` is expected to be a callable returning hidden states of shape
    (B, T, D) — see ``experiments/pipeline_demo.py`` for the backbone adapter.
    """
    optimizer.zero_grad()

    chosen_hidden = model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
    )
    B = chosen_hidden.shape[0]
    last_chosen_idx = (batch["chosen_input_ids"] != pad_id).sum(dim=-1) - 1
    last_chosen_hidden = chosen_hidden[torch.arange(B, device=chosen_hidden.device), last_chosen_idx, :]

    rejected_hidden = model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
    )
    B = rejected_hidden.shape[0]
    last_rejected_idx = (batch["rejected_input_ids"] != pad_id).sum(dim=-1) - 1
    last_rejected_hidden = rejected_hidden[torch.arange(B, device=rejected_hidden.device), last_rejected_idx, :]

    chosen_reward = reward_head_forward(last_chosen_hidden, reward_head.weight, reward_head.bias)
    rejected_reward = reward_head_forward(last_rejected_hidden, reward_head.weight, reward_head.bias)

    loss = pairwise_reward_loss(chosen_reward, rejected_reward)
    acc = pairwise_accuracy(chosen_reward, rejected_reward).item()

    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "accuracy": acc}


# --- Checkpoint persistence -----------------------------------------------------------
# A reward model = a (fine-tuned) causal-LM backbone + the scalar head weight/bias.
# The backbone is saved via HF ``save_pretrained`` (safetensors) so it reloads offline
# with ``AutoModelForCausalLM.from_pretrained``; the head + meta ride alongside it.

def save_reward_model(out_dir, backbone, reward_head, meta=None):
    """Persist RM backbone + scalar head + meta to ``out_dir``.

    Saves config + a plain ``state_dict`` rather than ``save_pretrained`` — the
    latter routes through accelerate/deepspeed's ``unwrap_model`` which crashes on
    clusters without CUDA_HOME. ``AutoModelForCausalLM.from_pretrained`` reads the
    config.json + pytorch_model.bin pair back fine (and torch.save handles GPT-2's
    tied weights, which bare safetensors would reject).
    """
    os.makedirs(out_dir, exist_ok=True)
    backbone.config.save_pretrained(out_dir)
    torch.save({k: v.detach().cpu() for k, v in backbone.state_dict().items()},
               os.path.join(out_dir, "pytorch_model.bin"))
    torch.save(
        {"weight": reward_head.weight.detach().cpu(),
         "bias": reward_head.bias.detach().cpu()},
        os.path.join(out_dir, "reward_head.pt"),
    )
    with open(os.path.join(out_dir, "reward_meta.json"), "w") as f:
        json.dump(meta or {}, f, indent=2)
    return out_dir


def load_reward_model(out_dir, device="cpu"):
    """Load a saved RM as a bundle ``{"model", "weight", "bias", "meta"}``.

    ``model`` is the frozen backbone (eval mode, ``output_hidden_states`` friendly);
    ``weight``/``bias`` are the raw head tensors consumed by ``reward_head_forward``.
    Shape matches the ``reward_bundle`` used by ``experiments/common.py`` scoring.
    """
    from transformers import AutoModelForCausalLM

    backbone = AutoModelForCausalLM.from_pretrained(out_dir).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = torch.load(os.path.join(out_dir, "reward_head.pt"), map_location=device)
    meta_path = os.path.join(out_dir, "reward_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return {"model": backbone, "weight": head["weight"].to(device),
            "bias": head["bias"].to(device), "meta": meta}
