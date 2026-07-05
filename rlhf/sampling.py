"""Batched, FSDP-safe sampling + masked log-prob recompute for GRPO rollouts.

Why not ``model.generate()``: under FSDP ``FULL_SHARD`` the HF generate path (KV cache,
internal reshapes) is fragile. A plain forward loop — ``logits = model(ids)`` each step —
lets FSDP all-gather params per layer transparently, so the exact same code runs single-GPU,
DDP, and FSDP. We trade the KV cache for robustness; for the benchmark's modest sequence
lengths that's fine.

Two functions:
  - ``sample_completions``: roll out ``n`` completions of one prompt (no grad).
  - ``gen_logprobs``: recompute summed log-probs over the *generated* tokens, with grad —
    this is the term GRPO differentiates. The mask guarantees we never train on prompt
    tokens (or, later, on tool-injected tokens).
"""
import torch
import torch.nn.functional as F


def _sample_next(logits, temperature, top_p):
    """Nucleus-sample one token per row. ``logits``: (B, V) -> (B,) token ids."""
    logits = logits / max(temperature, 1e-6)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Remove tokens once the cumulative prob (excluding the current one) exceeds p,
        # keeping at least the top token in every row.
        remove = (cumulative - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def sample_completions(model, prompt_ids, n, max_new_tokens,
                       eos_token_id, temperature=1.0, top_p=1.0, device=None,
                       sync_len=False):
    """Sample ``n`` completions of a single prompt with a manual forward loop.

    Args:
        model: the (possibly DDP/FSDP-wrapped) policy.
        prompt_ids: 1-D LongTensor of the prompt token ids.
        n: number of completions (the GRPO group size).
        eos_token_id: stop token.
        sync_len: if True, always run exactly ``max_new_tokens`` forwards (no
            data-dependent early stop). Required under FSDP, where every forward is a
            collective all-gather and all ranks must execute the *same* number of them
            — otherwise the ranks desync and NCCL aborts. Finished rows still emit EOS
            and are trimmed below, so output is identical either way.
    Returns:
        seqs: list of ``n`` 1-D LongTensors (prompt + generated, trimmed at first EOS).
        masks: list of ``n`` 1-D BoolTensors, True only on generated positions.
    """
    prompt_ids = prompt_ids.to(device)
    plen = prompt_ids.shape[-1]
    batch = prompt_ids.unsqueeze(0).repeat(n, 1)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        logits = model(input_ids=batch).logits[:, -1, :]
        nxt = _sample_next(logits, temperature, top_p)
        nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
        batch = torch.cat([batch, nxt.unsqueeze(-1)], dim=-1)
        finished = finished | (nxt == eos_token_id)
        if not sync_len and bool(finished.all()):
            break

    seqs, masks = [], []
    for i in range(n):
        gen = batch[i, plen:]
        eos_pos = (gen == eos_token_id).nonzero()
        end = int(eos_pos[0].item()) if len(eos_pos) else gen.shape[0]
        seq = batch[i, : plen + end].clone()
        mask = torch.zeros(seq.shape[0], dtype=torch.bool, device=device)
        mask[plen:] = True
        seqs.append(seq)
        masks.append(mask)
    return seqs, masks


def gen_logprobs(model, seqs, masks, pad_id, device):
    """Summed log-prob of the generated tokens in each sequence (differentiable).

    Pads the ragged batch, runs one forward, and for each sequence sums
    ``log p(token_t | token_<t)`` over the positions where ``mask`` is True.
    Returns a (B,) tensor aligned with ``seqs``.
    """
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    B = len(seqs)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, maxlen), dtype=torch.long, device=device)
    gmask = torch.zeros((B, maxlen), dtype=torch.bool, device=device)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, : lengths[i]] = s.to(device)
        attn[i, : lengths[i]] = 1
        gmask[i, : lengths[i]] = m.to(device)

    logits = model(input_ids=ids, attention_mask=attn).logits
    logprobs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    token_lp = logprobs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, maxlen-1)
    # A generated *target* at position t (t>=1) is scored from logits[t-1]; align the mask.
    target_mask = (gmask[:, 1:] & attn[:, 1:].bool()).float()
    return (token_lp * target_mask).sum(dim=-1)
