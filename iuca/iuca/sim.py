import numpy as np


def softmax(x, tau=1.0):
    x = x / max(tau, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class IterativeWorkingMemorySim:
    """
    Discrete simulator:
      - WM is an explicit set of K active items
      - pooled search score: s = M^T a
      - retain rK items, recruit (1-r)K
      - inhibition trace h to suppress repeats/lures
    """

    def __init__(self, N=300, K=16, r=0.8, tau=0.8,
                 inhibit_scale=1.0, kappa=0.95, gamma=1.0, seed=0):
        self.N = int(N)
        self.K = int(K)
        self.r = float(r)
        self.tau = float(tau)
        self.inhibit_scale = float(inhibit_scale)
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        # associative matrix (random symmetric baseline; replace with structured graph for experiments)
        A = self.rng.normal(0, 1, size=(self.N, self.N))
        self.M = (A + A.T) / 2.0
        self.M /= (np.linalg.norm(self.M, axis=0, keepdims=True) + 1e-9)

        self.a = np.zeros(self.N, dtype=np.float32)  # active indicator
        self.h = np.zeros(self.N, dtype=np.float32)  # inhibition trace

        init = self.rng.choice(self.N, size=self.K, replace=False)
        self.a[init] = 1.0

    def step(self, reset=False, inhibit_selected=False):
        K = self.K
        r = 0.0 if reset else self.r
        keep_n = int(round(r * K))
        add_n = K - keep_n

        W = np.where(self.a > 0)[0]
        assert len(W) == K

        # pooled multi-cue search
        s = self.M.T @ self.a

        # inhibition
        u = s - self.inhibit_scale * self.h

        # exclude current WM items from recruitment
        mask = np.ones(self.N, dtype=bool)
        mask[W] = False

        logits = np.full(self.N, -1e9, dtype=np.float32)
        logits[mask] = u[mask]
        p = softmax(logits, tau=self.tau)

        new_items = self.rng.choice(self.N, size=add_n, replace=False, p=p) if add_n > 0 else np.array([], dtype=int)

        keep_items = np.array([], dtype=int)
        if keep_n > 0:
            keep_scores = s[W]
            keep_items = W[np.argsort(-keep_scores)[:keep_n]]

        W_next = np.concatenate([keep_items, new_items])

        # update inhibition
        self.h *= self.kappa
        if inhibit_selected and len(new_items) > 0:
            self.h[new_items] += self.gamma

        # write next state
        self.a[:] = 0.0
        self.a[W_next] = 1.0

        overlap = len(set(W) & set(W_next)) / K
        return overlap, W_next
