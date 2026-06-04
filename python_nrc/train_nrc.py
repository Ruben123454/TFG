#!/usr/bin/env python3
"""
Train a PyTorch NRC MLP on samples dumped by the TFG transient renderer.

    python python_nrc/train_nrc.py
    python python_nrc/train_nrc.py --last-n 5000000 --epochs 30
"""
import argparse, sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_load_samples import load_samples   # noqa: E402
from nrc_model import NRCNet, save_model     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",         default="output/data/train_samples")
    ap.add_argument("--last-n",       type=int, default=None)
    ap.add_argument("--epochs",       type=int, default=60)
    ap.add_argument("--batch",        type=int, default=65536)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--val-frac",     type=float, default=0.1)
    ap.add_argument("--pos-freqs",    type=int, default=4)
    ap.add_argument("--t-freqs",      type=int, default=8)
    ap.add_argument("--hidden",       type=int, default=128)
    ap.add_argument("--layers",       type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "nrc_trained.pt")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d     = load_samples(args.base, last_n=args.last_n)
    scale = float(d["meta"].get("nrc_target_scale", 1.0))
    X     = torch.tensor(d["X"], dtype=torch.float32)
    y     = torch.tensor(d["y"], dtype=torch.float32)
    N     = X.shape[0]
    print(f"[data] {N} samples  scale={scale}  device={args.device}")

    g     = torch.Generator().manual_seed(0)
    perm  = torch.randperm(N, generator=g)
    n_val = int(N * args.val_frac)
    Xva, yva = X[perm[:n_val]].to(args.device), y[perm[:n_val]].to(args.device)
    Xtr, ytr = X[perm[n_val:]].to(args.device), y[perm[n_val:]].to(args.device)

    model = NRCNet(pos_freqs=args.pos_freqs, t_freqs=args.t_freqs,
                   hidden=args.hidden, layers=args.layers).to(args.device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    lossf = nn.MSELoss()
    print(f"[model] in_dim={model.in_dim}  hidden={args.hidden}x{args.layers}")

    ntr = Xtr.shape[0]
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(ntr, device=args.device)
        tot = 0.0
        for i in range(0, ntr, args.batch):
            idx = order[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vloss = lossf(model(Xva), yva).item()
            print(f"  epoch {ep+1:3d}  train MSE(log)={tot/ntr:.4e}  val MSE(log)={vloss:.4e}")

    save_model(args.out, model, d["meta"], scale)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
