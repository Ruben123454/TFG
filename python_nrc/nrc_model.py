"""
PyTorch NRC model for TFG transient branch.

Input layout (16 dims, already net-normalized):
  [0:3]  position xyz  (normalized to [0,1] by scene bounds)
  [3]    time t         (normalized to [0,1] by [t_min, t_max])
  [4:7]  direction      (already in [0,1])
  [7:10] normal         (already in [0,1])
  [10:16] albedo        (difuso rgb + especular rgb)

Output: 3 = log radiance = log(1 + L).  Decode: exp(x) - 1.
Scale = 1.0  (unlike TransientNRC which uses scale=16).
"""
import torch
import torch.nn as nn


def frequency_encode(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    if num_freqs <= 0:
        return x
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * torch.pi
    xb = x[..., None] * freqs
    enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
    enc = enc.reshape(*x.shape[:-1], -1)
    return torch.cat([x, enc], dim=-1)


class NRCNet(nn.Module):
    def __init__(self, pos_freqs=4, t_freqs=8, dir_freqs=2,
                 hidden=128, layers=5, out_dim=3,
                 use_dir=True, use_normal=True, use_albedo=True,
                 use_time=True, zero_init=True):
        super().__init__()
        self.cfg = dict(pos_freqs=pos_freqs, t_freqs=t_freqs, dir_freqs=dir_freqs,
                        hidden=hidden, layers=layers, out_dim=out_dim,
                        use_dir=use_dir, use_normal=use_normal, use_albedo=use_albedo,
                        use_time=use_time)

        def enc_dim(d, f): return d * (1 + 2*f) if f > 0 else d

        in_dim = enc_dim(3, pos_freqs)
        if use_time:
            in_dim += enc_dim(1, t_freqs)
        if use_dir:    in_dim += enc_dim(3, dir_freqs)
        if use_normal: in_dim += enc_dim(3, dir_freqs)
        if use_albedo: in_dim += 6

        mods, nin = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(nin, hidden), nn.ReLU(inplace=True)]
            nin = hidden
        mods += [nn.Linear(nin, out_dim)]
        self.net = nn.Sequential(*mods)
        self.in_dim = in_dim

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def encode(self, x16: torch.Tensor) -> torch.Tensor:
        parts = [frequency_encode(x16[..., 0:3], self.cfg["pos_freqs"])]
        if self.cfg.get("use_time", True):
            parts.append(frequency_encode(x16[..., 3:4], self.cfg["t_freqs"]))
        if self.cfg["use_dir"]:    parts.append(frequency_encode(x16[..., 4:7],  self.cfg["dir_freqs"]))
        if self.cfg["use_normal"]: parts.append(frequency_encode(x16[..., 7:10], self.cfg["dir_freqs"]))
        if self.cfg["use_albedo"]: parts.append(x16[..., 10:16])
        return torch.cat(parts, dim=-1)

    def forward(self, x16: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(x16))


def decode(log_pred: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Inverse of log(1 + scale*L).  TFG transient uses scale=1."""
    log_pred = torch.clamp(log_pred, min=0.0, max=80.0)
    return torch.clamp((torch.exp(log_pred) - 1.0) / scale, min=0.0, max=1000.0)


def save_model(path, model: NRCNet, meta: dict, scale: float = 1.0):
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg,
                "meta": meta, "scale": scale}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["cfg"])
    state = ckpt["state_dict"]
    target_in_dim = int(state["net.0.weight"].shape[1])

    model = NRCNet(**cfg).to(device)
    if model.in_dim != target_in_dim:
        legacy_cfg = dict(cfg)
        legacy_cfg.setdefault("t_freqs", 0)
        legacy_cfg["use_time"] = False
        legacy_model = NRCNet(**legacy_cfg).to(device)
        if legacy_model.in_dim == target_in_dim:
            model = legacy_model
            cfg = legacy_cfg
            print(f"[model] loaded legacy checkpoint without time input (in_dim={target_in_dim})")
        else:
            raise RuntimeError(
                f"checkpoint input dim {target_in_dim} does not match current cfg "
                f"({model.in_dim}) or legacy no-time cfg ({legacy_model.in_dim})"
            )

    model.load_state_dict(state)
    model.eval()
    return model, ckpt.get("meta", {}), float(ckpt.get("scale", 1.0))
