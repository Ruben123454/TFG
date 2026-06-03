#!/usr/bin/env python3
"""
Sample the TFG transient engine's trained C++ MLP (mlp_model.json) at Cornell
wall/floor/ceiling centers — pure torch reimplementation, no tinycudann needed.

TFG transient network (mlp.cpp):
  HashGrid(pos3+t, 4D, 16 levels, 2 feat/level, log2_hashmap=21, base=16, scale=1.5) -> 32 dims
  OneBlob(dir3+normal3, 4 bins)                                                        -> 24 dims
  Identity(difuso3+especular3) + 2 pad                                                ->  8 dims
  total encoding -> 64
  FullyFusedMLP(64 wide, 5 hidden, ReLU, no output activation)
  Target: log(1 + L)  scale=1

Usage:
    python python_nrc/sample_engine_mlp.py --no-data
    python python_nrc/sample_engine_mlp.py --last-n 2000000
    python python_nrc/sample_engine_mlp.py --selftest --last-n 500000
"""
import argparse, math, mmap, sys, warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_load_samples import load_samples   # noqa: E402
from nrc_model import decode                 # noqa: E402

PLOTS = REPO / "output" / "plots"

# ---- TFG transient network constants (must match mlp.cpp) ------------------
GRID_N_LEVELS        = 16
GRID_N_FEAT          = 2
GRID_LOG2_HASHMAP    = 21       # 4D grid, log2_hashmap=21
GRID_BASE_RES        = 16
GRID_PER_LEVEL_SCALE = 1.5
GRID_N_POS_DIMS      = 4        # pos3 + t
DIRNORM_BINS         = 4
MLP_WIDTH            = 64
MLP_HIDDEN_LAYERS    = 5
ENC_OUT = GRID_N_LEVELS * GRID_N_FEAT + 6 * DIRNORM_BINS + 6   # 62
ENC_PAD = ((ENC_OUT + 15) // 16) * 16                           # 64
HASH_PRIMES_4D = (1, 2654435761, 805459861, 3674653429)
MASK32 = (1 << 32) - 1


def _grid_layout():
    """Per-level (resolution, hashmap_size, offset) for the 4D HashGrid."""
    log2ps = np.float32(math.log2(np.float32(GRID_PER_LEVEL_SCALE)))
    res, hms, off = [], [], []
    offset = 0
    for L in range(GRID_N_LEVELS):
        scale = (np.float32(np.exp2(np.float32(np.float32(L) * log2ps)))
                 * np.float32(GRID_BASE_RES) - np.float32(1.0))
        r = int(math.ceil(float(scale))) + 1
        p = r ** GRID_N_POS_DIMS
        p = ((p + 7) // 8) * 8
        p = min(p, 1 << GRID_LOG2_HASHMAP)
        res.append(r); hms.append(p); off.append(offset)
        offset += p
    return res, hms, off, offset


def expected_n_params(grid_points):
    mlp = (MLP_WIDTH * ENC_PAD
           + (MLP_HIDDEN_LAYERS - 1) * MLP_WIDTH * MLP_WIDTH
           + 16 * MLP_WIDTH)
    return grid_points * GRID_N_FEAT + mlp, mlp


# ---- weight extraction (mmap, no full JSON parse) --------------------------
def json_has_key(json_path: Path, key: str) -> bool:
    with open(json_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return mm.find(b'"' + key.encode() + b'"') >= 0
        finally:
            mm.close()


def extract_fp16_block(json_path: Path, key: str) -> np.ndarray:
    cache = json_path.with_suffix(f".{key}.f16.npy")
    if cache.exists() and cache.stat().st_mtime >= json_path.stat().st_mtime:
        return np.load(cache)
    print(f"[weights] scanning {json_path.name} for '{key}' (caches to {cache.name})")
    with open(json_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            kpos = mm.find(b'"' + key.encode() + b'"')
            if kpos < 0:
                raise KeyError(f'"{key}" not found in {json_path}')
            bpos = mm.find(b'"bytes"', kpos)
            lb = mm.find(b'[', bpos); rb = mm.find(b']', lb)
            region = mm[lb + 1:rb]
        finally:
            mm.close()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = np.fromstring(region, dtype=np.uint8, sep=",")
    del region
    if raw.size % 2 != 0:
        raise ValueError(f"odd byte count for '{key}'")
    half = raw.view(np.float16)
    np.save(cache, half)
    return half


def extract_engine_weights(json_path: Path, weights: str = "auto"):
    if weights == "ema":
        if not json_has_key(json_path, "weights_ema_binary"):
            raise KeyError(f'"weights_ema_binary" not found in {json_path}')
        return extract_fp16_block(json_path, "weights_ema_binary"), "weights_ema_binary"
    if weights == "params":
        if not json_has_key(json_path, "params_binary"):
            raise KeyError(f'"params_binary" not found in {json_path}')
        return extract_fp16_block(json_path, "params_binary"), "params_binary"

    errors = []
    for key in ("weights_ema_binary", "params_binary"):
        if not json_has_key(json_path, key):
            errors.append(f'"{key}" not found in {json_path}')
            continue
        return extract_fp16_block(json_path, key), key

    raise KeyError(f"no supported fp16 weight block found in {json_path}: {'; '.join(errors)}")


# ---- engine network ---------------------------------------------------------
class EngineMLP:
    def __init__(self, params_fp16: np.ndarray, device: str):
        self.device = device
        self.res, self.hms, self.off, self.grid_points = _grid_layout()
        exp_total, mlp_count = expected_n_params(self.grid_points)
        if params_fp16.size != exp_total:
            raise ValueError(
                f"param count mismatch: file={params_fp16.size} expected={exp_total}.")

        p = torch.from_numpy(params_fp16.astype(np.float32))
        # NetworkWithInputEncoding: network params first, then encoding.
        w = p[:mlp_count]
        shapes = ([(MLP_WIDTH, ENC_PAD)]
                  + [(MLP_WIDTH, MLP_WIDTH)] * (MLP_HIDDEN_LAYERS - 1)
                  + [(16, MLP_WIDTH)])
        self.W, c = [], 0
        for (m, k) in shapes:
            self.W.append(w[c:c + m * k].reshape(m, k).to(device)); c += m * k
        self.grid = p[mlp_count:].reshape(self.grid_points, GRID_N_FEAT).to(device)

    @staticmethod
    def _qcdf(z, n):
        u = z * n; u2 = u * u
        return (((15./16.) * u * (1. - (2./3.)*u2 + (1./5.)*u2*u2) + 0.5).clamp(0., 1.))

    def _oneblob(self, x, n_bins):
        b = torch.arange(n_bins + 1, device=x.device, dtype=x.dtype) / n_bins
        z = b.view(1, 1, -1) - x.unsqueeze(-1)
        C = self._qcdf(z, n_bins) + self._qcdf(z - 1., n_bins) + self._qcdf(z + 1., n_bins)
        return (C[..., 1:] - C[..., :-1]).reshape(x.shape[0], -1)

    def _hash4d(self, cx, cy, cz, ct, hms):
        idx = ((cx * HASH_PRIMES_4D[0]) ^ (cy * HASH_PRIMES_4D[1])
               ^ (cz * HASH_PRIMES_4D[2]) ^ (ct * HASH_PRIMES_4D[3])) & MASK32
        return idx % hms

    def _grid(self, pos4):
        """pos4: (N,4) — [x,y,z,t] all in [0,1]."""
        N = pos4.shape[0]
        out = torch.zeros(N, GRID_N_LEVELS * GRID_N_FEAT, device=pos4.device)
        for L in range(GRID_N_LEVELS):
            log2ps = np.float32(math.log2(np.float32(GRID_PER_LEVEL_SCALE)))
            s = float(np.float32(np.exp2(np.float32(np.float32(L) * log2ps)))
                      * np.float32(GRID_BASE_RES) - np.float32(1.0))
            p  = pos4 * s + 0.5
            p0 = torch.floor(p)
            fr = p - p0
            p0i = p0.to(torch.int64)
            acc = torch.zeros(N, GRID_N_FEAT, device=pos4.device)
            for corner in range(16):   # 2^4 corners
                bx = (corner)      & 1
                by = (corner >> 1) & 1
                bz = (corner >> 2) & 1
                bt = (corner >> 3) & 1
                cx = p0i[:, 0] + bx; cy = p0i[:, 1] + by
                cz = p0i[:, 2] + bz; ct = p0i[:, 3] + bt
                wx = fr[:,0] if bx else (1. - fr[:,0])
                wy = fr[:,1] if by else (1. - fr[:,1])
                wz = fr[:,2] if bz else (1. - fr[:,2])
                wt = fr[:,3] if bt else (1. - fr[:,3])
                w  = (wx * wy * wz * wt).unsqueeze(-1)
                idx = self._hash4d(cx, cy, cz, ct, self.hms[L])
                acc = acc + w * self.grid[self.off[L] + idx]
            out[:, L * GRID_N_FEAT:(L + 1) * GRID_N_FEAT] = acc
        return out

    def encode(self, X16):
        pos4    = X16[:, 0:4].clamp(0., 1.)    # xyz + t
        dirnorm = X16[:, 4:10]
        alb     = X16[:, 10:16]
        N = X16.shape[0]
        return torch.cat([
            self._grid(pos4),
            self._oneblob(dirnorm, DIRNORM_BINS),
            alb,
            torch.ones(N, ENC_PAD - ENC_OUT, device=X16.device),
        ], dim=1)

    def forward(self, X16):
        h = self.encode(X16)
        for i, Wi in enumerate(self.W):
            h = h @ Wi.t()
            if i < len(self.W) - 1:
                h = torch.relu(h)
        return h[:, 0:3]

    @torch.no_grad()
    def luma(self, X16, scale=1.0):
        return decode(self.forward(X16), scale).mean(dim=1)


# ---- wall anchors -----------------------------------------------------------
def _load_meta_only(base):
    import json
    from plot_load_samples import _resolve
    j = _resolve(base).with_suffix(".json")
    if not j.exists():
        raise FileNotFoundError(f"{j} not found.")
    return json.loads(j.read_text())


def _norm_pos(meta, p):
    bmin = np.asarray(meta.get("bounds_min", [-6,-4,-10]), np.float32)
    bmax = np.asarray(meta.get("bounds_max", [6,6,2]),   np.float32)
    return np.clip((np.asarray(p,np.float32)-bmin)/np.maximum(bmax-bmin,1e-12),0.,1.)


def _map01(v): return np.clip((np.asarray(v,np.float32)+1.)/2., 0., 1.)


def _wall_anchors(meta):
    walls = [
        ("left wall",  (-3.,0.,-4.),  ( 1.,0.,0.), (0.1,0.8,0.9)),
        ("right wall", ( 3.,0.,-4.),  (-1.,0.,0.), (0.9,0.1,0.6)),
        ("floor",      ( 0.,-3.,-4.), ( 0.,1.,0.), (0.8,0.8,0.8)),
        ("ceiling",    ( 0., 3.,-4.), ( 0.,-1.,0.),(0.8,0.8,0.8)),
        ("back wall",  ( 0., 0.,-8.), ( 0.,0.,1.), (0.8,0.8,0.8)),
    ]
    anchors = []
    for name, raw, norm, kd in walls:
        x = np.zeros(16, np.float32)
        x[0:3]  = _norm_pos(meta, raw)
        x[3]    = 0.5                       # t: midpoint placeholder for sweep
        x[4:7]  = _map01(norm)
        x[7:10] = _map01(norm)
        x[10:13]= np.asarray(kd, np.float32)
        anchors.append(dict(name=name, raw_pos=np.asarray(raw,np.float32), x=x))
    return anchors


def _default_model():
    cands = [REPO/"build"/"mlp_model.json",
             REPO/"build"/"Release"/"mlp_model.json",
             REPO/"mlp_model.json"]
    cands = [c for c in cands if c.exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",   type=Path, default=_default_model())
    ap.add_argument("--weights", choices=["auto","ema","params"], default="auto")
    ap.add_argument("--base",    default="output/data/train_samples")
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--last-n",  type=int, default=None)
    ap.add_argument("--out",     type=Path, default=PLOTS/"nrc_engine_walls.png")
    ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.model or not Path(args.model).exists():
        ap.error("mlp_model.json not found; pass --model <path>")

    if args.no_data:
        meta = _load_meta_only(args.base); d = None
    else:
        d = load_samples(args.base, last_n=args.last_n); meta = d["meta"]

    scale = float(meta.get("nrc_target_scale", 1.0))
    half, key = extract_engine_weights(Path(args.model), args.weights)
    print(f"[weights] {key}: {half.size} fp16 params from {Path(args.model).name}")
    net = EngineMLP(half, args.device)
    print(f"[net] grid_points={net.grid_points}  enc={ENC_OUT}->{ENC_PAD}  "
          f"layers={len(net.W)}  (layout OK)")

    if args.selftest and d is not None:
        X  = torch.tensor(d["X"][:20000], dtype=torch.float32, device=args.device)
        yt = d["y_linear"][:20000].mean(1)
        with torch.no_grad():
            yp = net.luma(X, scale).cpu().numpy()
        corr = float(np.corrcoef(yp, yt)[0, 1])
        print(f"[selftest] corr(pred,target)={corr:.4f}")
        print(f"[selftest] pred [{yp.min():.3f},{yp.max():.3f}]  "
              f"target [{yt.min():.3f},{yt.max():.3f}]")
        return

    anchors = _wall_anchors(meta)
    print(f"[anchors] {len(anchors)} wall centers ready — run analyze_walls.py for full plot")


if __name__ == "__main__":
    main()
