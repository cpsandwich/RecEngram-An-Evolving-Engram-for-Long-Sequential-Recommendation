"""Loss functions for sequential recommendation training."""

import torch
import torch.nn.functional as F


def bce_loss_with_negatives(
    all_scores: torch.Tensor,
    l2_reg: float = 1e-5,
    model_params: list = None,
) -> torch.Tensor:
    """BCE loss with negative sampling (paper Eq. 11).

    Args:
        all_scores: (B, 1+N) — col 0 = positive, cols 1:N = negatives
        l2_reg: L2 regularization coefficient
        model_params: list of trainable parameters for L2
    """
    pos_logits = all_scores[:, 0]   # (B,)
    neg_logits = all_scores[:, 1:]  # (B, N)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction="mean"
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits), reduction="mean"
    )
    loss = pos_loss + neg_loss

    if l2_reg > 0 and model_params is not None:
        l2 = sum(p.pow(2).sum() for p in model_params if p.requires_grad)
        loss = loss + l2_reg * l2

    return loss


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return -torch.sigmoid(pos_scores - neg_scores).log().mean()


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    Args:
        anchor: (B, d)
        positive: (B, d)
        negatives: (B, N, d)
        temperature: softmax temperature (default 0.07 per SimCLR)
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    pos_sim = (anchor * positive).sum(dim=-1) / temperature  # (B,)
    neg_sim = torch.einsum("bd,bnd->bn", anchor, negatives) / temperature  # (B, N)

    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1+N)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
