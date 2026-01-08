import numpy as np


def make_random_graph(n_nodes: int = 200, degree: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        nbrs = rng.choice(n_nodes, size=degree, replace=False)
        adj[i, nbrs] = 1.0
        adj[nbrs, i] = 1.0
    np.fill_diagonal(adj, 0.0)
    # Row-normalize for transition probabilities
    row_sum = adj.sum(axis=1, keepdims=True) + 1e-9
    P = adj / row_sum
    return adj, P


def sample_walk(P: np.ndarray, start: int, length: int, rng: np.random.Generator):
    path = [start]
    cur = start
    for _ in range(length):
        cur = rng.choice(P.shape[0], p=P[cur])
        path.append(cur)
    return path  # length+1 nodes


def make_dataset(n_examples=5000, n_nodes=200, degree=6, min_len=2, max_len=4, seed=0):
    rng = np.random.default_rng(seed)
    adj, P = make_random_graph(n_nodes=n_nodes, degree=degree, seed=seed)

    X_start = np.zeros((n_examples,), dtype=np.int64)
    Y_next = np.zeros((n_examples,), dtype=np.int64)

    lengths = rng.integers(min_len, max_len + 1, size=n_examples)
    for idx in range(n_examples):
        s = int(rng.integers(0, n_nodes))
        L = int(lengths[idx])
        path = sample_walk(P, s, L, rng)
        # supervised target: predict first step given the start cue
        # (you can extend to multi-step supervision easily)
        X_start[idx] = path[0]
        Y_next[idx] = path[1]

    return {
        "adj": adj,
        "P": P,
        "X_start": X_start,
        "Y_next": Y_next,
        "n_nodes": n_nodes,
    }
