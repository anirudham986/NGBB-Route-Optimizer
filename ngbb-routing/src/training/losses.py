"""Loss functions for imitation learning and reinforcement learning.

Cross-entropy IL loss and REINFORCE policy gradient loss with KL penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def il_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Imitation learning cross-entropy loss.

    L_IL = -log P_θ(y* | G_t)

    Args:
        logits: Model output (batch, n_vars) raw scores.
        labels: Oracle labels (batch,) indices of chosen variables.

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(logits, labels)


def rl_policy_gradient_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: torch.Tensor,
    kl_penalty: float = 0.01,
    il_log_probs: torch.Tensor | None = None,
    entropy_bonus: float = 0.001,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """REINFORCE policy gradient loss with optional KL penalty.

    Args:
        log_probs: Log-probabilities of selected actions (batch,).
        rewards: Reward signals (batch,).
        baseline: Moving average baseline (batch,) or scalar.
        kl_penalty: Weight β for KL divergence against IL model.
        il_log_probs: Log-probs from frozen IL model for KL computation.
        entropy_bonus: Weight for entropy regularization.
        probs: Full probability distributions for entropy computation.

    Returns:
        Scalar loss tensor.
    """
    advantage = rewards - baseline
    pg_loss = -(log_probs * advantage.detach()).mean()

    loss = pg_loss

    # KL penalty against IL model
    if il_log_probs is not None and kl_penalty > 0:
        kl = (log_probs - il_log_probs).mean()
        loss = loss + kl_penalty * kl

    # Entropy bonus
    if probs is not None and entropy_bonus > 0:
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(-1).mean()
        loss = loss - entropy_bonus * entropy

    return loss


def kd_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Standalone Knowledge Distillation loss for Teacher->Student transfer.

    L_KD = α·T²·KL(σ(z_s/T) || σ(z_t/T)) + (1-α)·CE(z_s, y*)

    Args:
        student_logits: (batch, n_vars) student scores.
        teacher_logits: (batch, n_vars) teacher scores (detached).
        hard_labels: (batch,) oracle labels.
        temperature: Softmax temperature (higher = softer).
        alpha: Weight for soft-label loss vs hard-label.

    Returns:
        Scalar loss tensor.
    """
    T = temperature
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits.detach() / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher,
                         reduction='batchmean') * (T * T)
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def compute_top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    """Compute top-k accuracy for monitoring training.

    Args:
        logits: (batch, n_vars) scores.
        labels: (batch,) ground truth indices.
        k: Top-k to check.

    Returns:
        Accuracy as float in [0, 1].
    """
    topk = logits.topk(k, dim=-1).indices
    correct = (topk == labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()
