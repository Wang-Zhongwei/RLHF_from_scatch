"""From-scratch optimizer + schedule + gradient utilities.

``adamw_update`` reimplements decoupled-weight-decay AdamW so the pipeline does
not depend on ``torch.optim`` internals for the core update math.
"""
import torch


def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in-place AdamW step to ``param`` using ``grad`` and persistent ``state``."""
    if "step" not in state:
        state["step"] = 0
        state["m"] = torch.zeros_like(param)
        state["v"] = torch.zeros_like(param)

    state["step"] += 1
    t = state["step"]
    beta1, beta2 = betas

    state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
    state["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    m_hat = state["m"] / (1 - beta1 ** t)
    v_hat = state["v"] / (1 - beta2 ** t)

    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
    return state


def linear_warmup_schedule(step, warmup_steps):
    """Linear warmup multiplier in [0, 1]."""
    if warmup_steps == 0:
        return 1
    return min(1, step / warmup_steps)


def clip_grad_norm(grads, max_norm):
    """Rescale gradients in place so their global L2 norm is <= ``max_norm``."""
    total_norm = sum(g.pow(2).sum().item() for g in grads) ** 0.5
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for g in grads:
            g.mul_(scale)
    return float(total_norm)


def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    return torch.stack(grad_list, dim=0).mean(dim=0)
