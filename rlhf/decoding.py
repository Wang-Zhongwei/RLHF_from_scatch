"""Decoding: greedy / sampling / top-k / top-p, plus streaming and a chat wrapper."""
import torch

from .models import set_pad_token_to_eos


def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_input = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **encoded_input,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    return torch.argmax(logits, dim=-1).item()


def sample_with_temperature(logits, temperature):
    prob = (logits / temperature).softmax(dim=-1)
    return torch.multinomial(prob, num_samples=1).item()


def top_k_filter(logits, k):
    k = min(logits.shape[-1], k)
    vals, _ = logits.topk(k)
    threshold = vals[-1]
    return torch.where(logits < threshold, float("-inf"), logits)


def top_p_filter(logits, p):
    """Mask logits outside the smallest cumulative-probability nucleus of size p."""
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    n = logits.shape[-1]
    probs = logits.softmax(dim=-1)
    indices = probs.argsort(dim=-1, descending=True)

    k = (probs[indices].cumsum(dim=-1) < p).sum().item()
    logit_threshold = logits[indices[min(n - 1, k)]]

    return torch.where(logits < logit_threshold, float("-inf"), logits)


def stream_tokens(model, tokenizer, prompt, max_new_tokens):
    """Yield one decoded text piece per greedy-decoded new token."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1, :]
        next_token_id = greedy_decode(logits)
        yield tokenizer.decode([next_token_id], skip_special_tokens=True)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=-1)


def apply_stop_tokens(text, stop_tokens, eos_token):
    """Truncate text at the earliest occurrence of any stop token or the EOS token."""
    markers = list(stop_tokens) + ([eos_token] if eos_token is not None else [])
    earliest = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


def chat(model, tokenizer, user_message, system_prompt=None, max_new_tokens=32, stop_tokens=None):
    """Build a chat-style prompt, generate a reply, trim at stop tokens / EOS."""
    set_pad_token_to_eos(tokenizer)
    if max_new_tokens == 0:
        return ""
    prompt = (system_prompt + user_message) if system_prompt is not None else user_message
    text = generate_and_decode(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    reply = apply_stop_tokens(text, stop_tokens or [], tokenizer.eos_token)
    return reply.strip()
