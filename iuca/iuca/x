import torch
import torch.nn.functional as F


def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    # Stable-ish Gumbel(0,1)
    u = torch.rand_like(x).clamp_(1e-9, 1 - 1e-9)
    return -torch.log(-torch.log(u))


def gumbel_topk_straight_through(
    logits: torch.Tensor, k: int, tau: float = 1.0
):
    """
    Straight-through Gumbel-TopK selection.

    Args:
        logits: [B, N] or [N]
        k: number of picks
        tau: temperature for softmax

    Returns:
        hard_mask: [B, N] in {0,1} (exactly k ones per row if B>1)
        soft_probs: [B, N] probability distribution used for gradients
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    B, N = logits.shape
    k = int(k)
    k = max(0, min(k, N))

    if k == 0:
        hard = torch.zeros(B, N, device=logits.device, dtype=logits.dtype)
        soft = F.softmax(logits / max(tau, 1e-6), dim=-1)
        return hard, soft

    y = (logits + gumbel_noise_like(logits)) / max(tau, 1e-6)
    soft = F.softmax(y, dim=-1)

    topk_idx = torch.topk(soft, k=k, dim=-1).indices  # [B, k]
    hard = torch.zeros(B, N, device=logits.device, dtype=logits.dtype)
    hard.scatter_(1, topk_idx, 1.0)

    # Straight-through estimator
    hard_st = hard + (soft - soft.detach())
    return hard_st, soft
