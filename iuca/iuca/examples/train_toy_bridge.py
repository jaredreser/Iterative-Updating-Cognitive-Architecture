import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from iuca.torch_module import IterativeUpdatingWorkingMemory, InhibitionTrace, RetentionSchedule
from benchmarks.bridge_retrieval import make_dataset


class OneStepBridgeModel(nn.Module):
    """
    Wraps IUCA with:
      - learnable node embeddings as candidate pool
      - WM initialized from the start cue embedding
      - output logits over next-node candidates from pooled query
    """
    def __init__(self, n_nodes: int, d: int = 64, K: int = 8):
        super().__init__()
        self.n_nodes = n_nodes
        self.d = d
        self.K = K

        self.embed = nn.Embedding(n_nodes, d)

        self.iuca = IterativeUpdatingWorkingMemory(d=d, K=K, tau=0.8, inhibit_scale=1.0, use_meaning_shift=True)

        # Predictor head from pooled query to node logits
        self.head = nn.Linear(d, n_nodes)

        self.inhib = InhibitionTrace(kappa=0.95, gamma=0.5)

    def forward(self, start_ids, h, r: float):
        """
        start_ids: [B]
        h: [B, N]
        """
        B = start_ids.shape[0]
        C = self.embed.weight.unsqueeze(0).expand(B, self.n_nodes, self.d)  # [B, N, d]

        # init WM: repeat the start embedding across K slots (simple baseline)
        s = self.embed(start_ids)  # [B, d]
        W = s.unsqueeze(1).expand(B, self.K, self.d).contiguous()

        W_next, keep_probs, select_probs, aux = self.iuca(W, C, h=h, r=r)

        # pooled query for output: reuse aux["q"] (already pooled)
        q = aux["q"]  # [B, d]
        logits = self.head(q)  # [B, N]

        # update inhibition based on selected distribution (you can also inhibit only on errors)
        h_next = self.inhib(h, select_probs.detach())

        return logits, h_next, {"keep": keep_probs, "select": select_probs, "q": q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--degree", type=int, default=6)
    ap.add_argument("--examples", type=int, default=8000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--use_curriculum", action="store_true")
    args = ap.parse_args()

    ds = make_dataset(n_examples=args.examples, n_nodes=args.nodes, degree=args.degree, seed=0)
    X = torch.tensor(ds["X_start"], dtype=torch.long)
    Y = torch.tensor(ds["Y_next"], dtype=torch.long)
    N = ds["n_nodes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneStepBridgeModel(n_nodes=N, d=args.d, K=args.K).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    schedule = RetentionSchedule(r_min=0.25, r_max=0.85, s0=2000, s1=1200)

    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        Xp, Yp = X[perm], Y[perm]

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for i in range(0, len(Xp), args.batch):
            xb = Xp[i:i+args.batch].to(device)
            yb = Yp[i:i+args.batch].to(device)

            B = xb.shape[0]
            h = torch.zeros(B, N, device=device)

            r = schedule(step) if args.use_curriculum else 0.75

            logits, h_next, aux = model(xb, h, r=r)

            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == yb).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1
            step += 1

        print(f"epoch {epoch+1}/{args.epochs}  loss={total_loss/n_batches:.4f}  acc={total_acc/n_batches:.3f}")

    print("Done.")


if __name__ == "__main__":
    main()
