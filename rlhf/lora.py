"""LoRA adapters: low-rank weight deltas, parameter freezing, and merging."""
import torch
import torch.nn.functional as F


def lora_delta(A, B, alpha, r):
    """Scaled low-rank weight update: (alpha/r) * B @ A."""
    return (alpha / r) * (B @ A)


def lora_linear_forward(x, base_weight, A, B, alpha, r, bias=None):
    effective_weight = base_weight + lora_delta(A, B, alpha, r)
    return F.linear(x, effective_weight, bias)


def init_lora_weights(in_features, out_features, r, seed=0):
    """Return (A, B) with random A and zero B so the initial delta is zero."""
    torch.manual_seed(seed)
    A = torch.randn(r, in_features, dtype=torch.float32) * 0.01
    B = torch.zeros(out_features, r, dtype=torch.float32)
    return A, B


def freeze_base_params(model):
    """Freeze everything except LoRA adapter params."""
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(base_weight, lora_a, lora_b, scaling):
    """Fold the scaled low-rank update B @ A back into the base weight."""
    return base_weight + scaling * lora_b @ lora_a
