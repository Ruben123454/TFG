#!/usr/bin/env python3
"""
Load raw training samples dumped by the TFG transient renderer
(output/data/train_samples.f32 + .json).

Each row is 19 float32:
  [pos_x,pos_y,pos_z, t, dir_x,dir_y,dir_z, n_x,n_y,n_z,
   kd_r,kd_g,kd_b, ks_r,ks_g,ks_b, target_log_r,target_log_g,target_log_b]

  - pos (0:3): raw world-space; normalize with bounds_min/bounds_max from JSON
  - t   (3):   raw seconds; normalize with t_min/t_max from JSON
  - dir (4:7), normal (7:10): already in [0,1]
  - albedo (10:16): raw [0,1]
  - target (16:19): log(1 + L)  — decode with exp(x)-1

Library:
    from plot_load_samples import load_samples
    d = load_samples()   # d["X"] (N,16) normalized, d["y"] (N,3) log target

Script:
    python python_nrc/plot_load_samples.py
    python python_nrc/plot_load_samples.py --train
"""
import argparse, json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_T_MIN = 0.0
DEFAULT_T_MAX = 9.8e-8 * 1.01
COLS = ["pos_x","pos_y","pos_z","t",
        "dir_x","dir_y","dir_z",
        "n_x","n_y","n_z",
        "kd_r","kd_g","kd_b",
        "ks_r","ks_g","ks_b",
        "target_log_r","target_log_g","target_log_b"]


def _unique_ints(values):
    out = []
    for v in values:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv > 0 and iv not in out:
            out.append(iv)
    return out


def _infer_floats_per_row(meta, total_floats):
    columns = meta.get("columns", [])
    count = int(meta.get("count", 0) or 0)
    count_width = (total_floats // count) if count > 0 and total_floats % count == 0 else None
    candidates = _unique_ints([
        meta.get("floats_per_row", None),
        len(columns) if isinstance(columns, list) else None,
        count_width,
        len(COLS),
        19,
    ])

    for c in candidates:
        if c >= len(COLS) and total_floats % c == 0:
            requested = int(meta.get("floats_per_row", c) or c)
            if requested != c:
                print(f"[load] warning: metadata floats_per_row={requested}, "
                      f"but file size matches {c}; using {c}")
            return c

    raise ValueError(
        f"cannot infer train sample row width: {total_floats} float32 values "
        f"do not divide any supported row size {candidates}. "
        "Expected the current 19-float NRC dump."
    )


def _resolve(base):
    base = Path(base)
    if base.suffix in (".f32", ".json"):
        base = base.with_suffix("")
    if not base.with_suffix(".f32").exists():
        hits = list(REPO.rglob(base.name + ".f32"))
        if hits:
            base = max(hits, key=lambda p: p.stat().st_mtime).with_suffix("")
    return base


def load_samples(base="output/data/train_samples", last_n=None):
    base = _resolve(base)
    data_path = base.with_suffix(".f32")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found (run a training pass first).")

    meta = {}
    if base.with_suffix(".json").exists():
        meta = json.loads(base.with_suffix(".json").read_text())

    data_bytes = data_path.stat().st_size
    if data_bytes % 4 != 0:
        raise ValueError(f"{data_path} size is not a multiple of float32 bytes.")
    total_floats = data_bytes // 4
    floats = _infer_floats_per_row(meta, total_floats)
    row_bytes = floats * 4
    total_rows = data_bytes // row_bytes

    if total_rows == 0:
        raise ValueError(f"{data_path} is empty — run the renderer first.")

    if last_n and 0 < last_n < total_rows:
        with open(data_path, "rb") as f:
            f.seek((total_rows - last_n) * row_bytes)
            raw = np.fromfile(f, dtype="<f4", count=last_n * floats).reshape(-1, floats)
        print(f"[load] last {last_n} of {total_rows} rows")
    else:
        raw = np.fromfile(data_path, dtype="<f4").reshape(-1, floats)

    X = raw[:, 0:16].copy()

    # Normalize pos and t to [0,1] (matches the CUDA input kernel).
    bmin = np.asarray(meta.get("bounds_min", [0,0,0]), dtype=np.float32)
    bmax = np.asarray(meta.get("bounds_max", [1,1,1]), dtype=np.float32)
    span = np.maximum(bmax - bmin, 1e-12)
    pos_n = (X[:, 0:3] - bmin) / span
    pos_oob = ((pos_n < 0) | (pos_n > 1)).any(axis=1)
    X[:, 0:3] = np.clip(pos_n, 0.0, 1.0)

    if "t_min" not in meta or "t_max" not in meta:
        print(f"[load] warning: metadata has no t_min/t_max; using renderer default "
              f"[{DEFAULT_T_MIN}, {DEFAULT_T_MAX}] seconds")
    tmin = float(meta.get("t_min", DEFAULT_T_MIN))
    tmax = float(meta.get("t_max", DEFAULT_T_MAX))
    X[:, 3] = np.clip((X[:, 3] - tmin) / max(tmax - tmin, 1e-12), 0.0, 1.0)

    y = raw[:, 16:19].copy()
    scale = float(meta.get("nrc_target_scale", 1.0))
    y_linear = np.maximum(0.0, (np.exp(y) - 1.0) / scale)

    return {
        "raw": raw, "X": X, "y": y, "y_linear": y_linear,
        "meta": meta, "columns": COLS,
        "raw_bounds_min": raw[:, 0:3].min(axis=0),
        "raw_bounds_max": raw[:, 0:3].max(axis=0),
        "pos_oob_fraction": float(pos_oob.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base",   default="output/data/train_samples")
    ap.add_argument("--last-n", type=int, default=None)
    ap.add_argument("--train",  action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    d = load_samples(args.base, last_n=args.last_n)
    N = d["raw"].shape[0]
    scale = float(d["meta"].get("nrc_target_scale", 1.0))
    print(f"[loaded] {N} samples  floats/row={d['raw'].shape[1]}  scale={scale}")
    print(f"         bounds_min={d['meta'].get('bounds_min')}  bounds_max={d['meta'].get('bounds_max')}")
    print(f"         t=[{d['meta'].get('t_min')}, {d['meta'].get('t_max')}]")
    print(f"         raw_pos_min={d['raw_bounds_min']}  raw_pos_max={d['raw_bounds_max']}")
    print(f"         pos outside bounds: {d['pos_oob_fraction']*100:.2f}%")
    print(f"         target log luma: mean={d['y'].mean(1).mean():.4f}  max={d['y'].mean(1).max():.4f}")
    print(f"         target linear luma: mean={d['y_linear'].mean(1).mean():.4g}  max={d['y_linear'].mean(1).max():.4g}")

    if args.train:
        import torch, torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        X = torch.tensor(d["X"], dtype=torch.float32, device=dev)
        y = torch.tensor(d["y"], dtype=torch.float32, device=dev)
        mods, nin = [], 16
        for _ in range(4):
            mods += [nn.Linear(nin, 128), nn.ReLU()]; nin = 128
        mods += [nn.Linear(128, 3)]
        net = nn.Sequential(*mods).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        n, bs = X.shape[0], min(65536, X.shape[0])
        print(f"[train] {n} samples  device={dev}  epochs={args.epochs}")
        for ep in range(args.epochs):
            perm = torch.randperm(n, device=dev); tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i+bs]; opt.zero_grad()
                loss = nn.functional.mse_loss(net(X[idx]), y[idx])
                loss.backward(); opt.step(); tot += loss.item()*len(idx)
            if ep == 0 or (ep+1) % 5 == 0 or ep == args.epochs-1:
                print(f"  epoch {ep+1:3d}  MSE(log)={tot/n:.4e}")


if __name__ == "__main__":
    main()
