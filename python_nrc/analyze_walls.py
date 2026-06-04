#!/usr/bin/env python3
"""
For each Cornell wall/floor/ceiling center, plot energy vs time:
  - gray dots   : nearby training samples  (x=t, y=luma)
  - orange line : binned average of those samples
  - blue line   : PyTorch MLP  L(t) sweep  (--torch-model)
  - magenta line: C++ engine MLP L(t) sweep (--engine-model)

Usage:
    python python_nrc/analyze_walls.py

    python python_nrc/analyze_walls.py --torch-model python_nrc/nrc_trained.pt

    python python_nrc/analyze_walls.py --engine-model build/mlp_model.json

    python python_nrc/analyze_walls.py \\
        --torch-model  python_nrc/nrc_trained.pt \\
        --engine-model build/mlp_model.json

    python python_nrc/analyze_walls.py --last-n 5000000
"""
import argparse, sys, math
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_load_samples import load_samples                        # noqa: E402
from nrc_model import decode, load_model as load_torch_model     # noqa: E402

PLOTS = REPO / "output" / "plots"
COLOR_DATA_DOTS = "#8c8c8c"
COLOR_DATA_AVG = "#e66100"
COLOR_TORCH = "#0072b2"
COLOR_ENGINE = "#d81b60"

WALLS = [
    ("left wall",  (-3., 0., -4.),  ( 1., 0., 0.), (0.1, 0.8, 0.9)),
    ("right wall", ( 3., 0., -4.),  (-1., 0., 0.), (0.9, 0.1, 0.6)),
    ("floor",      ( 0.,-3., -4.),  ( 0., 1., 0.), (0.8, 0.8, 0.8)),
    ("ceiling",    ( 0., 3., -4.),  ( 0.,-1., 0.), (0.8, 0.8, 0.8)),
    ("back wall",  ( 0., 0., -8.),  ( 0., 0., 1.), (0.8, 0.8, 0.8)),
]


def _norm_pos(meta, p):
    bmin = np.asarray(meta.get("bounds_min", [-6,-4,-10]), np.float32)
    bmax = np.asarray(meta.get("bounds_max", [6, 6, 2]),  np.float32)
    return np.clip((np.asarray(p, np.float32) - bmin) / np.maximum(bmax-bmin, 1e-12), 0., 1.)


def _map01(v): return np.clip((np.asarray(v, np.float32) + 1.) / 2., 0., 1.)


def _anchor_base(meta, raw_pos, normal, kd):
    """16-dim input row for this wall center; t (col 3) will be swept."""
    x = np.zeros(16, np.float32)
    x[0:3]  = _norm_pos(meta, raw_pos)
    x[3]    = 0.5          # placeholder — overwritten during sweep
    x[4:7]  = _map01(normal)
    x[7:10] = _map01(normal)
    x[10:13]= np.asarray(kd, np.float32)
    return x


def _binned_avg(t, y, bins=64):
    b      = np.clip((t * bins).astype(np.int32), 0, bins-1)
    sums   = np.bincount(b, weights=y, minlength=bins)
    counts = np.bincount(b, minlength=bins)
    good   = counts > 0
    cx     = (np.arange(bins, dtype=np.float32) + 0.5) / bins
    avg    = np.zeros(bins, np.float32)
    avg[good] = sums[good] / counts[good]
    return cx[good], avg[good]


def _load_engine(model_path, device, weights):
    from sample_engine_mlp import EngineMLP, extract_engine_weights
    half, key = extract_engine_weights(Path(model_path), weights)
    return EngineMLP(half, device), key


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base",         default="output/data/train_samples")
    ap.add_argument("--last-n",       type=int, default=None)
    ap.add_argument("--pos-eps",      type=float, default=0.02)
    ap.add_argument("--normal-eps",   type=float, default=0.1)
    ap.add_argument("--max-pts",      type=int, default=3000)
    ap.add_argument("--avg-bins",     type=int, default=64)
    ap.add_argument("--sweep-steps",  type=int, default=256)
    ap.add_argument("--torch-model",  type=Path, default=None)
    ap.add_argument("--engine-model", type=Path, default=None)
    ap.add_argument("--engine-weights", choices=["auto", "ema", "params"], default="auto")
    ap.add_argument("--out", type=Path, default=PLOTS / "nrc_wall_analysis.png")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- load samples ----
    d     = load_samples(args.base, last_n=args.last_n)
    X     = d["X"]      # (N,16) normalized — col 3 is t
    meta  = d["meta"]
    scale = float(meta.get("nrc_target_scale", 1.0))
    pos   = X[:, 0:3]
    nrm   = X[:, 7:10]
    tcol  = X[:, 3]
    y_lin = decode(torch.tensor(d["y"]), scale).numpy().mean(1)
    print(f"[data] {X.shape[0]} samples  scale={scale}")

    # ---- load models ----
    torch_net = None
    if args.torch_model and args.torch_model.exists():
        torch_net, _, torch_scale = load_torch_model(args.torch_model, args.device)
        print(f"[torch]  loaded {args.torch_model.name}  scale={torch_scale}")
    else:
        torch_scale = scale

    engine_net = None
    engine_key = None
    if args.engine_model and args.engine_model.exists():
        engine_net, engine_key = _load_engine(args.engine_model, args.device, args.engine_weights)
        print(f"[engine] loaded {args.engine_model.name} ({engine_key})")
    elif args.engine_model:
        print(f"[engine] warning: {args.engine_model} not found; skipping C++ MLP curve")

    # ---- plot ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(3, len(WALLS))
    rows = math.ceil(len(WALLS) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.8*rows), squeeze=False)
    rng  = np.random.default_rng(0)
    ts   = np.linspace(0., 1., args.sweep_steps, dtype=np.float32)

    for wi, (name, raw_pos, normal, kd) in enumerate(WALLS):
        ax    = axes[wi // cols][wi % cols]
        ref   = _norm_pos(meta, raw_pos)
        n_ref = _map01(normal)

        d_pos = np.linalg.norm(pos - ref[None, :], axis=1)
        d_nrm = np.linalg.norm(nrm - n_ref[None, :], axis=1)
        idx   = np.where((d_pos < args.pos_eps) & (d_nrm < args.normal_eps))[0]
        n_near = len(idx)

        if n_near == 0:
            ax.text(0.5, 0.5, "no nearby samples\n(increase --pos-eps)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
        else:
            plot_idx = (idx if len(idx) <= args.max_pts
                        else rng.choice(idx, args.max_pts, replace=False))
            ax.scatter(tcol[plot_idx], y_lin[plot_idx],
                       s=5, alpha=0.28, color=COLOR_DATA_DOTS, zorder=2,
                       label=f"data ({n_near} near, {len(plot_idx)} shown)")
            bx, by = _binned_avg(tcol[idx], y_lin[idx], args.avg_bins)
            ax.plot(bx, by, color=COLOR_DATA_AVG, lw=1.9, label="data avg", zorder=3)

        # Build sweep: fix all inputs, vary only t (col 3)
        base  = _anchor_base(meta, raw_pos, normal, kd)
        sweep = np.repeat(base[None, :], args.sweep_steps, axis=0)
        sweep[:, 3] = ts

        if torch_net is not None:
            with torch.no_grad():
                yp = decode(torch_net(torch.tensor(sweep, device=args.device)),
                            torch_scale).cpu().numpy().mean(1)
            ax.plot(ts, yp, color=COLOR_TORCH, lw=1.9, label="PyTorch MLP", zorder=5)

        if engine_net is not None:
            with torch.no_grad():
                ye = engine_net.luma(
                    torch.tensor(sweep, dtype=torch.float32, device=args.device),
                    scale).cpu().numpy()
            engine_label = "C++ MLP (EMA)" if engine_key == "weights_ema_binary" else "C++ MLP (params)"
            ax.plot(ts, ye, color=COLOR_ENGINE, lw=1.9, label=engine_label, zorder=4)

        rp = np.asarray(raw_pos)
        ax.set_title(f"{name}  ({rp[0]:.1f},{rp[1]:.1f},{rp[2]:.1f})", fontsize=9)
        ax.set_xlabel("t (normalized)")
        ax.set_ylabel("linear luma")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for j in range(len(WALLS), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(
        f"L(t) at wall/floor/ceiling centers  "
        f"pos_eps={args.pos_eps}  last_n={args.last_n or 'all'}",
        fontsize=11)
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
