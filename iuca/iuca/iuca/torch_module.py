from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .selection import gumbel_topk_straight_through


@dataclass
class RetentionSchedule:
    """
    Simple logistic schedule for r(step).

    r(step) = r_min + (r_max - r_min) * sigmoid((step - s0)/s1)
    """
    r_min: float = 0.2
    r_max: float = 0.85
    s0: float = 10_000.0
    s1: float = 3_000.0

    def __call__(self, step: int) -> float:
        x = (float(step) - self.s0) / max(self.s1, 1e-6)
        sig = 1.0 / (1.0 + math.exp(-x))
        return self.r_min + (self.r_max - self.r_min) * sig


class InhibitionTrace(nn.Module):
    """
    Iterative inhibition: h <- kappa*h + gamma*z
    where z is a (soft) selection indicator over candidates.
    """
    def __init__(self, kappa: float = 0.95, gamma: float = 1.0):
        super().__init__()
        self.kappa = float(kappa)
        self.gamma = float(gamma)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h, z: [B, N]
        return self.kappa * h + self.gamma * z


class IterativeUpdatingWorkingMemory(nn.Module):
    """
    Iterative-Updating Working Memory (IWM / IUCA) module.

    Key mechanisms embodied:
      - Capacity-limited working set W: [B, K, d]
      - Pooled multi-cue search: q = pool(W), candidate logits ~ sim(C, q)
      - Retain–drop–add: keep rK slots; recruit (1-r)K candidates
      - Iterative inhibition via h: subtract inhibit_scale*h from logits
      - Optional "meaning shift": context-conditioned slot transform before pooling

    Inputs:
      W: [B, K, d]  working memory slots
      C: [B, N, d]  candidate pool
      h: [B, N]     inhibition trace (optional)
      r: float      retention fraction override (optional)
      step: int     for retention schedule (optional)

    Outputs:
      W_next: [B, K, d]
      keep_mask: [B, K] (soft)
      select_mask: [B, N] (soft; used for inhibition update)
      aux: dict with useful diagnostics
    """
    def __init__(
        self,
        d: int,
        K: int,
        tau: float = 0.8,
        inhibit_scale: float = 1.0,
        use_meaning_shift: bool = True,
    ):
        super().__init__()
        self.d = int(d)
        self.K = int(K)
        self.tau = float(tau)
        self.inhibit_scale = float(inhibit_scale)
        self.use_meaning_shift = bool(use_meaning_shift)

        # Pooling: mean(W) -> q
        self.pool = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Keep scoring: slot + q -> scalar
        self.keep_mlp = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        # Optional meaning shift: slot + q -> delta(slot)
        if self.use_meaning_shift:
            self.shift = nn.Sequential(
                nn.LayerNorm(2 * d),
                nn.Linear(2 * d, d),
                nn.GELU(),
                nn.Linear(d, d),
            )

    def forward(self, W, C, h=None, r: float | None = None):
        B, K, d = W.shape
        _, N, _ = C.shape
        assert K == self.K and d == self.d, "Shape mismatch vs module config."

        if h is None:
            h = torch.zeros(B, N, device=W.device, dtype=W.dtype)

        # --- pooled multi-cue query q ---
        q = self.pool(W.mean(dim=1))  # [B, d]

        # Optional meaning shift / contextual remapping within WM
        if self.use_meaning_shift:
            q_rep = q.unsqueeze(1).expand(B, K, d)
            delta = self.shift(torch.cat([W, q_rep], dim=-1))
            W_eff = W + delta
        else:
            W_eff = W

        # Recompute q from shifted WM for tighter coupling
        q = self.pool(W_eff.mean(dim=1))  # [B, d]

        # --- candidate logits via pooled similarity ---
        logits = torch.einsum("bnd,bd->bn", C, q) / math.sqrt(d)
        logits = logits - self.inhibit_scale * h

        # --- keep / recruit counts ---
        if r is None:
            r = 0.75
        r = float(max(0.0, min(1.0, r)))

        keep_n = int(round(r * K))
        add_n = K - keep_n

        # --- choose kept slots (soft -> top-k hard-ish) ---
        q_rep = q.unsqueeze(1).expand(B, K, d)
        keep_logits = self.keep_mlp(torch.cat([W_eff, q_rep], dim=-1)).squeeze(-1)  # [B, K]
        keep_mask_st, keep_probs = gumbel_topk_straight_through(keep_logits, k=keep_n, tau=self.tau)  # [B, K]

        # Gather kept slots (compact them)
        if keep_n > 0:
            keep_idx = torch.topk(keep_probs, k=keep_n, dim=-1).indices  # [B, keep_n]
            kept = torch.gather(W, 1, keep_idx.unsqueeze(-1).expand(B, keep_n, d))
        else:
            kept = W[:, :0, :]  # empty [B, 0, d]

        # --- recruit new candidates ---
        # NOTE: This does not enforce "no duplicates" w.r.t. kept items; in practice, candidate pools are large,
        # and you can optionally mask candidates that correspond to currently held items if you maintain IDs.
        select_mask_st, select_probs = gumbel_topk_straight_through(logits, k=add_n, tau=self.tau)  # [B, N]

        # Recruited vectors: [B, add_n, d]
        # Convert selection mask to explicit picks for clarity
        if add_n > 0:
            pick_idx = torch.topk(select_probs, k=add_n, dim=-1).indices  # [B, add_n]
            new = torch.gather(C, 1, pick_idx.unsqueeze(-1).expand(B, add_n, d))
        else:
            new = C[:, :0, :]

        W_next = torch.cat([kept, new], dim=1)  # [B, K, d]

        aux = {
            "q": q,
            "keep_logits": keep_logits,
            "cand_logits": logits,
            "keep_n": keep_n,
            "add_n": add_n,
        }
        return W_next, keep_probs, select_probs, aux
