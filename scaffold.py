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
def set_pad_token_to_eos(tokenizer):
    # TODO: assign tokenizer.pad_token = tokenizer.eos_token and return the tokenizer
    tokenizer.pad_token =  tokenizer.eos_token
    return tokenizer

# ── Step 004  generate_and_decode ──
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_input = tokenizer(prompt, return_tensors='pt')
    output_ids = model.generate(**encoded_input, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

# ── Step 005  greedy_decode ──
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    # TODO: return the token id with the largest logit as a Python int
    return torch.argmax(logits, dim=-1).item()

# ── Step 006  sample_with_temperature ──
def sample_with_temperature(logits, temperature):
    prob = (logits / temperature).softmax(dim=-1)
    return torch.multinomial(prob, num_samples=1).item()

# ── Step 007  top_k_filter ──
def top_k_filter(logits, k):
    k = min(logits.shape[-1], k)
    vals, _ = logits.topk(k)
    threshold = vals[-1]
    return torch.where(logits < threshold, float('-inf'), logits)

# ── Step 008  top_p_filter ──
def top_p_filter(logits, p):
    # TODO: mask logits outside the smallest cumulative-probability nucleus of size p.
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    
    n = logits.shape[-1]
    probs = logits.softmax(dim=-1)
    indices = probs.argsort(dim=-1, descending=True)

    k = (probs[indices].cumsum(dim=-1) < p).sum().item()
    logit_threshold = logits[indices[min(n-1, k)]]

    return torch.where(logits < logit_threshold, float('-inf'), logits)

# ── Step 009  build_synthetic_instruction_dataset ──
def build_synthetic_instruction_dataset():
    return [
        {
            "prompt": "What is the capital of France?", 
            "response": "The capital of France is Paris."
        },
        {
            "prompt": "Write a Python function to add two numbers.", 
            "response": "def add(a, b):\n    return a + b"
        },
        {
            "prompt": "Who wrote Romeo and Juliet?", 
            "response": "William Shakespeare wrote Romeo and Juliet."
        },
        {
            "prompt": "Explain what a neural network is.", 
            "response": "A neural network is a machine learning model inspired by the human brain."
        }
    ]

# ── Step 010  format_example ──
def format_example(example):
    """
    Renders a single instruction example into one training string.
    """
    return f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"

# ── Step 011  apply_template ──
def apply_template(examples):
    # TODO: apply format_example to each item in examples and return the list of strings.
    return [
        format_example(example) for example in examples
    ]

# ── Step 012  tokenize_example ──
def tokenize_example(tokenizer, text, max_length=64):
    # TODO: encode `text` with truncation at max_length, no padding, return list[int]
    return tokenizer.encode(text, padding=False, max_length=max_length)

# ── Step 013  build_labels ──
def build_labels(input_ids):
    # TODO: return a fresh list equal to input_ids to serve as next-token labels
    return list(input_ids)

# ── Step 014  mask_prompt_labels ──
def mask_prompt_labels(labels, prompt_length):
    # Create a new list where the first 'prompt_length' items are -100
    return [-100 if i < prompt_length else label for i, label in enumerate(labels)]

# ── Step 015  pad_batch ──
def pad_batch(sequences, pad_id):
    # TODO: right-pad a list of token id sequences to the longest length using pad_id
    n_max = max(len(seq) for seq in sequences)
    return [seq + [pad_id]*(n_max - len(seq)) for seq in sequences]

# ── Step 016  make_attention_mask ──
def make_attention_mask(padded_ids, pad_id):
    # Return a same-shape 0/1 mask with 1 where token != pad_id else 0
    return [[1 if token != pad_id else 0 for token in row] for row in padded_ids]

# ── Step 017  collate_lm_batch ──
def collate_lm_batch(batch, pad_id):
    input_ids = [x['input_ids'] for x in batch]
    labels = [x['labels'] for x in batch]

    input_ids = torch.tensor(pad_batch(input_ids, pad_id))
    labels = torch.tensor(pad_batch(labels, -100))

    attention_mask = torch.where(input_ids == pad_id, 0, 1)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }

# ── Step 018  iterate_minibatches ──
import random

def iterate_minibatches(examples, batch_size, seed=0):
    # TODO: yield shuffled minibatches of size batch_size from examples (deterministic per seed).
    rng = random.Random(seed)

    n = len(examples)
    rng.shuffle(examples)

    start = 0
    while start < n:
        end = min(n, start + batch_size)
        yield examples[start:end]
        start = end

# ── Step 019  train_val_split ──
import random
import math

def train_val_split(examples, val_ratio=0.2, seed=0):
    rng = random.Random(seed)

    shuffled_examples = list(examples) # shallow copy is enough
    n = len(shuffled_examples)

    rng.shuffle(shuffled_examples)

    val_size = math.floor(val_ratio * n)
    train_size = n - val_size

    return shuffled_examples[:train_size], shuffled_examples[train_size:]

# ── Step 020  shift_logits_and_labels ──
def shift_logits_and_labels(logits, labels):
    # TODO: drop the last logit position and the first label position so token t predicts t+1
    return logits[:, :-1, :], labels[:, 1:]

# ── Step 021  cross_entropy_loss ──
import torch
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next-token cross-entropy, ignoring label positions equal to -100."""
    # TODO: reduce (B, T-1, V) logits and (B, T-1) labels to a scalar loss tensor.
    log_probs = shift_logits.softmax(dim=-1).log()
    mask = (shift_labels == -100)
    safe_labels = shift_labels.clone()
    safe_labels[mask] = 0

    target_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(dim=-1))
    loss = -target_log_probs[~mask].mean()

    return loss

# ── Step 022  adamw_update ──
import torch

def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in-place AdamW step to `param` using `grad` and persistent `state`."""
    # Initialize state on first call
    if 'step' not in state:
        state['step'] = 0
        state['m'] = torch.zeros_like(param)
        state['v'] = torch.zeros_like(param)

    # Increment step counter
    state['step'] += 1
    t = state['step']

    # Extract beta values
    beta1, beta2 = betas

    # Update biased first moment estimate
    state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)

    # Update biased second raw moment estimate
    state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Compute bias-corrected moment estimates
    m_hat = state['m'] / (1 - beta1 ** t)
    v_hat = state['v'] / (1 - beta2 ** t)

    # Apply decoupled weight decay
    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    # Apply adaptive update step
    param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

    return state

# ── Step 023  linear_warmup_schedule ──
def linear_warmup_schedule(step, warmup_steps):
    # TODO: return a linear warmup multiplier in [0, 1] given the current step and warmup window.
    if warmup_steps == 0:
        return 1
    else:
        return min(1, step/warmup_steps)

# ── Step 024  clip_grad_norm ──
import torch

def clip_grad_norm(grads, max_norm):
    # Compute global L2 norm across all gradient tensors
    total_norm = sum(g.pow(2).sum().item() for g in grads) ** 0.5

    # Rescale in place if norm exceeds max_norm
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for g in grads:
            g.mul_(scale)

    return float(total_norm)

# ── Step 025  accumulate_gradients ──
import torch

def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    # TODO: average a list of equally-shaped gradient tensors and return the mean tensor
    return torch.stack(grad_list, dim=0).mean(dim=0)

# ── Step 026  sft_train_step ──
import torch

def sft_train_step(model, batch, optimizer):
    """Run one SFT forward/backward/step and return the loss as a float."""
    # Clear gradients before backprop
    optimizer.zero_grad()

    # Forward pass
    logits = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    ).logits

    # Shift logits and labels for causal LM
    shifted_logits, shifted_labels = shift_logits_and_labels(logits, batch['labels'])

    # Compute cross-entropy loss
    loss = cross_entropy_loss(shifted_logits, shifted_labels)

    # Backprop and optimizer step
    loss.backward()
    optimizer.step()

    return loss.item()

# ── Step 027  evaluate_loss ──
import torch

def evaluate_loss(model, batches):
    """Mean LM loss over validation batches, no grad."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in batches:
            logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            ).logits
            shifted_logits, shifted_labels = shift_logits_and_labels(logits, batch['labels'])
            loss = cross_entropy_loss(shifted_logits, shifted_labels)
            total_loss += loss.item()
            count += 1

    return total_loss / count

# ── Step 028  lora_delta ──
import torch

def lora_delta(A, B, alpha, r):
    # Scaled low-rank weight update: (alpha/r) * B @ A
    return (alpha / r) * (B @ A)

# ── Step 029  lora_linear_forward ──
import torch
import torch.nn.functional as F

def lora_linear_forward(x, base_weight, A, B, alpha, r, bias=None):
    # Compute effective weight = base + LoRA delta
    effective_weight = base_weight + lora_delta(A, B, alpha, r)
    return F.linear(x, effective_weight, bias)

# ── Step 030  init_lora_weights ──
import torch

def init_lora_weights(in_features, out_features, r, seed=0):
    """Return (A, B) LoRA factors with random A and zero B so the initial delta is zero."""
    torch.manual_seed(seed)
    A = torch.randn(r, in_features, dtype=torch.float32) * 0.01
    B = torch.zeros(out_features, r, dtype=torch.float32)
    return A, B

# ── Step 031  freeze_base_params ──
import torch

def freeze_base_params(model):
    # Freeze all params except LoRA adapter params
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    return model

# ── Step 032  count_trainable_params ──
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Step 033  merge_lora ──
def merge_lora(base_weight, lora_a, lora_b, scaling):
    # TODO: fold the scaled low-rank update B @ A back into the base weight matrix.
    return base_weight + scaling * lora_b @ lora_a

# ── Step 034  build_synthetic_preference_dataset ──
import random

def build_synthetic_preference_dataset(num_examples=8, seed=0):
    pool = [
        {"prompt": "What is the boiling point of water?",
         "chosen": "Water boils at 100 degrees Celsius at sea level.",
         "rejected": "Water boils when it gets hot enough."},
        {"prompt": "Who wrote Romeo and Juliet?",
         "chosen": "Romeo and Juliet was written by William Shakespeare.",
         "rejected": "I am not sure who wrote it."},
        {"prompt": "What is 2 + 2?",
         "chosen": "2 + 2 equals 4.",
         "rejected": "2 + 2 equals 5."},
        {"prompt": "What does HTTP stand for?",
         "chosen": "HTTP stands for HyperText Transfer Protocol.",
         "rejected": "HTTP is a computer term."},
        {"prompt": "What is the speed of light?",
         "chosen": "The speed of light in a vacuum is approximately 3x10^8 meters per second.",
         "rejected": "Light travels very fast."},
        {"prompt": "What is DNA?",
         "chosen": "DNA is a molecule that carries genetic instructions for living organisms.",
         "rejected": "DNA is found in cells."},
        {"prompt": "What is the capital of France?",
         "chosen": "The capital of France is Paris.",
         "rejected": "I do not know."},
        {"prompt": "What is 2 + 2?",
         "chosen": "2 + 2 equals 4.",
         "rejected": "2 + 2 equals 5."},
    ]
    rng = random.Random(seed)
    if num_examples <= len(pool):
        return rng.sample(pool, num_examples)
    else:
        indices = list(range(len(pool))) * (num_examples // len(pool) + 1)
        rng.shuffle(indices)
        return [pool[i] for i in indices[:num_examples]]

# ── Step 035  format_preference ──
def format_preference(example):
        return {
                'chosen_text': example['prompt'] + ' ' + example['chosen'],
                        'rejected_text': example['prompt'] + ' ' + example['rejected']
                            }

# ── Step 036  reward_head_forward ──
import torch
def reward_head_forward(hidden_state, weight, bias):
    """Map a final hidden state to a scalar reward via a linear projection."""
    # flatten weight to (D,) to handle both (D,) and (1, D) inputs
    w = weight.view(-1)
    return hidden_state @ w + bias

# ── Step 037  pairwise_reward_loss ──
import torch
import torch.nn.functional as F
def pairwise_reward_loss(chosen_reward, rejected_reward):
    """Bradley-Terry pairwise loss: mean(-log_sigmoid(chosen - rejected))."""
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()

# ── Step 038  reward_bce_loss ──
import numpy as np
def reward_bce_loss(chosen_reward, rejected_reward):
    # softplus(-r_c) = -log(sigmoid(r_c)), softplus(r_r) = -log(1 - sigmoid(r_r))
    chosen_loss = np.logaddexp(0, -chosen_reward)
    rejected_loss = np.logaddexp(0, rejected_reward)
    return float(np.mean(0.5 * (chosen_loss + rejected_loss)))

# ── Step 039  pairwise_accuracy ──
import torch

def pairwise_accuracy(chosen_reward, rejected_reward):
    """Fraction of pairs where chosen_reward > rejected_reward."""
    # TODO: return the fraction of pairs where chosen strictly beats rejected
    return (chosen_reward > rejected_reward).mean(dtype=float)

# ── Step 040  reward_train_step ──
import torch

def reward_train_step(model, reward_head, batch, optimizer):
    # TODO: forward chosen+rejected, score last token, compute loss/acc, step optimizer
    optimizer.zero_grad()

    pad_id = 0
    chosen_hidden_states = model(
        input_ids=batch['chosen_input_ids'], # (B, T)
        attention_mask=batch['chosen_attention_mask']
    ) # (B, T, D)
    B, T, D = chosen_hidden_states.shape

    last_chosen_idx = (batch['chosen_input_ids'] != pad_id).sum(dim=-1) - 1 # (B,)
    # 1. use gather 
    # last_chosen_idx.view(B, 1, 1).expand(B, 1, D)
    # last_chosen_hidden_state = chosen_hidden_states.gather(dim=1, last_chosen_idx).squeeze(1) # (B, D)

    # 2. use advanced indexing
    last_chosen_hidden_state = chosen_hidden_states[torch.arange(B), last_chosen_idx, :]

    rejected_hidden_states = model(
        input_ids=batch['rejected_input_ids'],
        attention_mask=batch['rejected_attention_mask']
    )
    B, T, D = rejected_hidden_states.shape

    last_rejected_idx = (batch['rejected_input_ids'] != pad_id).sum(dim=-1) - 1 # (B,)
    last_rejected_hidden_state = rejected_hidden_states[torch.arange(B), last_rejected_idx, :]

    chosen_reward = reward_head_forward(last_chosen_hidden_state, reward_head.weight, reward_head.bias)
    rejected_reward = reward_head_forward(last_rejected_hidden_state, reward_head.weight, reward_head.bias)

    loss = pairwise_reward_loss(chosen_reward, rejected_reward)
    acc = pairwise_accuracy(chosen_reward, rejected_reward).item()

    loss.backward()
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'accuracy': acc
    }

# ── Step 041  sequence_logprob ──
import torch
import torch.nn.functional as F

def sequence_logprob(logits, token_ids):
    """Sum log probabilities of the selected tokens along the sequence dimension."""
    # TODO: return a scalar tensor equal to sum_t log_softmax(logits)[t, token_ids[t]]
    B = token_ids.shape[0]
    return logits.log_softmax(dim=-1)[torch.arange(B), token_ids].sum()

# ── Step 042  per_token_kl ──
import numpy as np

def per_token_kl(policy_logprobs, ref_logprobs):
    """Per-token KL estimate between policy and reference log-probs."""
    # TODO: return the per-token KL contribution used in the PPO penalty
    return policy_logprobs - ref_logprobs

# ── Step 043  compute_returns ──
import numpy as np

def compute_returns(rewards, gamma=0.99):
    """Return the discounted return at each timestep as a 1D numpy array."""
    # TODO: turn a per-timestep reward sequence into discounted returns
    T = len(rewards)
    returns = np.zeros(T)

    for t in reversed(range(T)):
        if t == T - 1:
            returns[t] = rewards[t]
        else:
            returns[t] = rewards[t] + returns[t + 1] * gamma
    
    return returns

# ── Step 044  gae_advantages ──
def gae_advantages(rewards, values, gamma=0.99, lam=0.95):
    # TODO: compute GAE advantages of shape (T,) from rewards (T,) and values (T+1,)
    T = len(rewards)
    adv = np.zeros(T+1)

    for t in reversed(range(T)):
        delta_t = rewards[t] + gamma * values[t+1] - values[t]
        adv[t] = delta_t + gamma * lam * adv[t+1]
    
    return adv[:-1]

# ── Step 045  policy_ratio ──
import torch

def policy_ratio(new_logprobs, old_logprobs):
    """Return the PPO importance ratio exp(new - old) elementwise."""
    # TODO: exponentiate the difference between new and old log probabilities
    return (new_logprobs - old_logprobs).exp()

# ── Step 046  clipped_surrogate ──
import torch

def clipped_surrogate(ratio, advantages, clip_eps=0.2):
    """PPO clipped surrogate loss (scalar tensor to minimize)."""
    # TODO: combine ratio and advantages via the PPO clipped objective and return a scalar loss
    return -torch.min(ratio * advantages, torch.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages).mean()

# ── Step 047  value_function_loss ──
import torch

def value_function_loss(values, returns):
    """Mean squared error between predicted values and target returns."""
    # TODO: compute mean((values - returns) ** 2) as a scalar tensor
    return (values - returns).pow(2).mean()

# ── Step 048  entropy_bonus ──
import torch

def entropy_bonus(logits):
    """Return mean categorical entropy of the distribution defined by `logits` over the last axis."""
    # TODO: softmax over the vocab axis, compute -sum(p * log p), then average.
    probs = logits.softmax(dim=-1)
    return -(probs * probs.log()).sum(dim=-1).mean()

# ── Step 049  ppo_loss ──
import torch

def ppo_loss(ratio, advantages, values, returns, logits, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    # TODO: combine clipped surrogate, value loss, and entropy bonus into the full PPO loss dict.
    policy_loss = clipped_surrogate(ratio, advantages, clip_eps)
    value_loss = value_function_loss(values, returns)
    entropy = entropy_bonus(logits)
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'loss': loss
    }

# ── Step 050  kl_penalized_reward ──
import torch

def kl_penalized_reward(reward, kl, beta=0.1):
    """Return reward shaped by a KL penalty against a reference policy."""
    # TODO: combine the reward model score with a beta-weighted KL penalty
    return reward - beta * kl

# ── Step 051  batch_sequence_logprob ──
import torch
import torch.nn.functional as F

def batch_sequence_logprob(logits, token_ids, attention_mask=None):
    log_probs = logits.log_softmax(dim=-1)
    B, T, V = log_probs.shape
    # attention_mask (B, T)
    per_token = log_probs[torch.arange(B).unsqueeze(1), torch.arange(T).unsqueeze(0), token_ids]
    if attention_mask is not None:
        return (per_token * attention_mask).sum(dim=-1)
    else:
        return per_token.sum(dim=-1)

# ── Step 052  dpo_logratios ──
import torch

def dpo_logratios(policy_chosen_logps, policy_rejected_logps):
    """Return policy_chosen_logps - policy_rejected_logps elementwise."""
    # TODO: compute the policy log-ratio used inside the DPO objective
    return policy_chosen_logps - policy_rejected_logps

# ── Step 053  dpo_ref_logratios ──
import torch

def dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps):
    # TODO: return per-example chosen minus rejected reference log probabilities
    return ref_chosen_logps - ref_rejected_logps

# ── Step 054  dpo_loss ──
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Return the DPO loss as a scalar torch tensor."""
    # TODO: combine policy and reference log-ratios into the DPO log-sigmoid loss
    r = dpo_logratios(policy_chosen_logps, policy_rejected_logps)
    r_ref = dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps)

    return -F.logsigmoid(beta * (r - r_ref)).mean()

# ── Step 055  ipo_loss ──
import torch

def ipo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # TODO: regress (policy_logratios - ref_logratios) toward the IPO target 1/(2*beta)
    h = dpo_logratios(policy_chosen_logps, policy_rejected_logps) - dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps)
    return (h - 1/(2*beta)).pow(2).mean()

# ── Step 056  kto_loss ──
import torch

def kto_loss(policy_logps, ref_logps, labels, beta=0.1):
    # TODO: implement KTO loss for unpaired desirable/undesirable examples.
    r = beta * (policy_logps - ref_logps)
    r *= torch.where(labels == 1, 1, -1)
    return (1 - r.sigmoid()).mean()

# ── Step 057  orpo_loss ──
import torch
import torch.nn.functional as F

def orpo_loss(policy_chosen_logps, policy_rejected_logps, sft_loss, lambda_or=0.1):
    # TODO: return sft_loss + lambda_or * mean(-log_sigmoid(log_odds_chosen - log_odds_rejected))
    # convert log-probs to log-odds: log(p/(1-p)) = log_p - log(1 - exp(log_p))
    log_odds_chosen = policy_chosen_logps - torch.log1p(-policy_chosen_logps.exp())
    log_odds_rejected = policy_rejected_logps - torch.log1p(-policy_rejected_logps.exp())
    return sft_loss + lambda_or * (-F.logsigmoid(log_odds_chosen - log_odds_rejected)).mean()

# ── Step 058  simpo_loss ──
import torch
import torch.nn.functional as F

def simpo_loss(policy_chosen_logps, policy_rejected_logps, chosen_lengths, rejected_lengths, beta=2.0, gamma=0.5):
    """Return the mean SimPO loss as a scalar tensor."""
    # TODO: form length-normalized implicit rewards and apply the beta/gamma margin loss
    chosen_reward = beta * policy_chosen_logps / chosen_lengths
    rejected_reward = beta * policy_rejected_logps / rejected_lengths
    return (-F.logsigmoid(chosen_reward - rejected_reward - gamma)).mean()

# ── Step 059  build_eval_prompt_set ──
def build_eval_prompt_set():
    # TODO: return a held-out list of at least 4 short instruction-style eval prompts
    return [
        "Explain what a neural network is in simple terms.",
        "Write a short poem about the ocean.",
        "What are three tips for staying productive while working from home?",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "What is the difference between supervised and unsupervised learning?",
        "Give me a recipe for a simple pasta dish.",
    ]

# ── Step 060  generate_completions ──
def generate_completions(model, tokenizer, prompts, max_new_tokens=16):
    """Return a list of greedy completions, one per prompt."""
    # TODO: produce one decoded completion per prompt, preserving input order
    return [generate_and_decode(model, tokenizer, prompt, max_new_tokens=max_new_tokens) for prompt in prompts]

# ── Step 061  score_with_reward ──
import torch

def score_with_reward(reward_model, tokenizer, prompt, completion):
    """Return a scalar reward float for the prompt+completion pair."""
    # TODO: tokenize prompt+completion, run the backbone, apply the reward head.
    inputs = tokenizer(prompt + completion, return_tensors='pt')
    with torch.no_grad():
        outputs = reward_model['model'](**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1][0, -1, :]  # (D,)
    score = reward_head_forward(last_hidden_state, reward_model['weight'], reward_model['bias'])
    return score.item()

# ── Step 062  win_rate ──
def win_rate(reward_model, tokenizer, prompts, completions_a, completions_b):
    """Fraction of prompts where A's completion outscores B's under the reward model.

    Ties count as 0.5. Returns a float in [0, 1].
    """
    a_win_cnt = 0
    for prompt, completion_a, completion_b in zip(prompts, completions_a, completions_b):
        score_a = score_with_reward(reward_model, tokenizer, prompt, completion_a)
        score_b = score_with_reward(reward_model, tokenizer, prompt, completion_b)
        a_win_cnt += (1 if score_a > score_b else 0 if score_a < score_b else 0.5)
    
    return a_win_cnt / len(prompts)

# ── Step 063  stream_tokens ──
import torch

def stream_tokens(model, tokenizer, prompt, max_new_tokens):
    # TODO: yield one decoded text piece per greedy-decoded new token, up to max_new_tokens.
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    for i in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1, :]  # (V,)
        next_token_id = greedy_decode(logits)
        piece = tokenizer.decode([next_token_id], skip_special_tokens=True)
        yield piece
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=-1)

# ── Step 064  apply_stop_tokens ──
def apply_stop_tokens(text, stop_tokens, eos_token):
    # TODO: truncate text at the earliest occurrence of any stop token or the eos token
    markers = list(stop_tokens) + ([eos_token] if eos_token is not None else [])
    earliest = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]

# ── Step 065  chat ──
def chat(model, tokenizer, user_message, system_prompt=None, max_new_tokens=32, stop_tokens=None):
    # TODO: build a chat-style prompt, generate a reply, and trim it at stop tokens / EOS.
    set_pad_token_to_eos(tokenizer)
    if max_new_tokens == 0:
        return ''
    prompt = (system_prompt + user_message) if system_prompt is not None else user_message
    text = generate_and_decode(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    eos_token = tokenizer.eos_token
    reply = apply_stop_tokens(text, stop_tokens or [], eos_token)
    return reply.strip()

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
