import torch
import torch.nn.functional as F


def overlap_from_keep_mask(keep_probs: torch.Tensor, keep_n: int) -> torch.Tensor:
    """
    Approx overlap as fraction of slots retained (batchwise).
    keep_probs: [B, K] (soft)
    """
    B, K = keep_probs.shape
    return torch.full((B,), float(keep_n) / float(K), device=keep_probs.device)


def soft_overlap(W: torch.Tensor, W_next: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between pooled summaries of W and W_next.
    W, W_next: [B, K, d]
    """
    a = W.mean(dim=1)
    b = W_next.mean(dim=1)
    return F.cosine_similarity(a, b, dim=-1)


def reset_rate(overlaps: torch.Tensor, threshold: float = 0.2) -> float:
    """
    overlaps: [T] or [T,B] tensor
    """
    x = overlaps
    if x.dim() == 2:
        x = x.mean(dim=1)
    return float((x < threshold).float().mean().item())


def lure_reselect_rate(selected_hist: torch.Tensor, window: int = 5) -> float:
    """
    Rough lure trapping proxy:
    selected_hist: [T, B, N] selection probabilities or hard masks.
    Counts how often top-1 selection repeats within a short window.
    """
    T, B, N = selected_hist.shape
    top = selected_hist.argmax(dim=-1)  # [T,B]
    repeats = 0
    total = 0
    for t in range(T):
        for dt in range(1, window + 1):
            if t - dt >= 0:
                repeats += (top[t] == top[t - dt]).float().sum().item()
                total += B
    return float(repeats / max(total, 1))
