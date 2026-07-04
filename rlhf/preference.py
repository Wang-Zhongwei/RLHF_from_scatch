"""Direct preference-optimization objectives: DPO, IPO, KTO, ORPO, SimPO."""
import torch
import torch.nn.functional as F


def batch_sequence_logprob(logits, token_ids, attention_mask=None):
    """Sum per-token log-probs over the sequence, optionally masking padding."""
    log_probs = logits.log_softmax(dim=-1)
    B, T, V = log_probs.shape
    per_token = log_probs[
        torch.arange(B).unsqueeze(1), torch.arange(T).unsqueeze(0), token_ids
    ]
    if attention_mask is not None:
        return (per_token * attention_mask).sum(dim=-1)
    return per_token.sum(dim=-1)


def dpo_logratios(policy_chosen_logps, policy_rejected_logps):
    return policy_chosen_logps - policy_rejected_logps


def dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps):
    return ref_chosen_logps - ref_rejected_logps


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO log-sigmoid loss over policy vs reference log-ratios."""
    r = dpo_logratios(policy_chosen_logps, policy_rejected_logps)
    r_ref = dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps)
    return -F.logsigmoid(beta * (r - r_ref)).mean()


def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """IPO: regress the log-ratio gap toward the target 1/(2*beta)."""
    h = (dpo_logratios(policy_chosen_logps, policy_rejected_logps)
         - dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps))
    return (h - 1 / (2 * beta)).pow(2).mean()


def kto_loss(policy_logps, ref_logps, labels, beta=0.1):
    """KTO loss for unpaired desirable(+1)/undesirable(-1) examples."""
    r = beta * (policy_logps - ref_logps)
    r = r * torch.where(labels == 1, 1, -1)
    return (1 - r.sigmoid()).mean()


def orpo_loss(policy_chosen_logps, policy_rejected_logps, sft_loss, lambda_or=0.1):
    """ORPO: SFT loss + odds-ratio preference term (reference-free)."""
    log_odds_chosen = policy_chosen_logps - torch.log1p(-policy_chosen_logps.exp())
    log_odds_rejected = policy_rejected_logps - torch.log1p(-policy_rejected_logps.exp())
    return sft_loss + lambda_or * (-F.logsigmoid(log_odds_chosen - log_odds_rejected)).mean()


def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               chosen_lengths, rejected_lengths, beta=2.0, gamma=0.5):
    """SimPO: length-normalized implicit rewards with a beta/gamma margin."""
    chosen_reward = beta * policy_chosen_logps / chosen_lengths
    rejected_reward = beta * policy_rejected_logps / rejected_lengths
    return (-F.logsigmoid(chosen_reward - rejected_reward - gamma)).mean()
