# Video ViT Encoder/Decoder w/ Temporal Unit (compact) + MAE visible-token support
# - keep_idx ONLY (no keep_mask / no -1 padding): Nvis is constant across frames
# - No attention masks passed into SDPA (tokens are physically dropped via gather)
# - 2D RoPE uses original patch coordinates via rope_pos_idx gather
# - Temporal attention: (global CLS + per-frame CLS) over time

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- utils -----------------------------

def trunc_normal_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return t.normal_(0.0, std).clamp_(-2 * std, 2 * std)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, L, D)
    cos/sin: (L, D) or (B, L, D)
    """
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    elif cos.dim() == 3:
        cos, sin = cos[:, None], sin[:, None]
    else:
        raise ValueError(f"cos/sin must be (L,D) or (B,L,D), got {tuple(cos.shape)}")

    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[..., 0::2], sin[..., 0::2]
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    return torch.stack((y1, y2), dim=-1).flatten(-2)


def rope1d_cos_sin(d: int, L: int, base: float, device, dtype):
    assert d % 2 == 0
    half = d // 2
    inv = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(L, device=device, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]
    cos = ang.cos().repeat_interleave(2, dim=-1)
    sin = ang.sin().repeat_interleave(2, dim=-1)
    return cos.to(dtype), sin.to(dtype)


def rope2d_cos_sin(d: int, h: int, w: int, base: float, device, dtype):
    # d split into (x,y), requires d%4==0; output (N=h*w, d)
    assert d % 4 == 0
    q = d // 4
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    inv = 1.0 / (base ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    ax = xs.reshape(-1)[:, None] * inv[None, :]
    ay = ys.reshape(-1)[:, None] * inv[None, :]

    cosx = ax.cos().repeat_interleave(2, dim=-1)
    sinx = ax.sin().repeat_interleave(2, dim=-1)
    cosy = ay.cos().repeat_interleave(2, dim=-1)
    siny = ay.sin().repeat_interleave(2, dim=-1)
    cos = torch.cat((cosx, cosy), dim=-1)
    sin = torch.cat((sinx, siny), dim=-1)
    return cos.to(dtype), sin.to(dtype)


# ----------------------------- modules -----------------------------

class PatchEmbed(nn.Module):
    def __init__(self, patch: int, in_ch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B*, C, H, W) -> (B*, N, D), hw=(h,w)
        x = self.proj(x)
        hw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, hw


class MHSA(nn.Module):
    def __init__(self, dim: int, heads: int, rope_base: float, rope_kind: str):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.base = rope_base
        self.kind = rope_kind  # "none" | "1d" | "2d"
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        self._shape = None  # (L,) or (H,W)

    def _ensure_rope(self, shape, device, dtype):
        if self.kind == "none":
            return
        if (
            self._cos is not None
            and self._shape == shape
            and self._cos.device == device
            and self._cos.dtype == dtype
        ):
            return
        if self.kind == "1d":
            L = int(shape)
            self._cos, self._sin = rope1d_cos_sin(self.d, L, self.base, device, dtype)
        else:
            h, w = shape
            self._cos, self._sin = rope2d_cos_sin(self.d, h, w, self.base, device, dtype)
        self._shape = shape

    def forward(
        self,
        x: torch.Tensor,                       # (B, L, D)
        *,
        rope_shape=None,
        rope_pos_idx: Optional[torch.Tensor] = None,  # (B, L-n_prefix) or (L-n_prefix,)
        n_prefix: int = 0,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, L, d)

        if self.kind != "none" and n_prefix < L:
            self._ensure_rope(rope_shape, x.device, x.dtype)
            cos_full, sin_full = self._cos, self._sin

            if rope_pos_idx is None:
                cos = cos_full[: (L - n_prefix)]
                sin = sin_full[: (L - n_prefix)]
            elif rope_pos_idx.dim() == 1:
                cos = cos_full.index_select(0, rope_pos_idx)
                sin = sin_full.index_select(0, rope_pos_idx)
            elif rope_pos_idx.dim() == 2:
                flat = rope_pos_idx.reshape(-1)
                cos = cos_full.index_select(0, flat).view(B, L - n_prefix, self.d)
                sin = sin_full.index_select(0, flat).view(B, L - n_prefix, self.d)
            else:
                raise ValueError(f"rope_pos_idx must be 1D or 2D, got {tuple(rope_pos_idx.shape)}")

            q = torch.cat((q[:, :, :n_prefix], apply_rope(q[:, :, n_prefix:], cos, sin)), dim=2)
            k = torch.cat((k[:, :, :n_prefix], apply_rope(k[:, :, n_prefix:], cos, sin)), dim=2)

        y = F.scaled_dot_product_attention(
            q * self.scale, k, v,
            attn_mask=None,          # IMPORTANT: no masks (we drop tokens with keep_idx)
            dropout_p=0.0,
            is_causal=False,
        )
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, 2 * hid, bias=True)  # SwiGLU
        self.fc2 = nn.Linear(hid, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


class VideoBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, rope_base: float, eps: float):
        super().__init__()
        self.n_spa = nn.RMSNorm(dim, eps=eps)
        self.spa = MHSA(dim, heads, rope_base, rope_kind="2d")  # RoPE on patches

        self.n_tmp = nn.RMSNorm(dim, eps=eps)
        self.tmp = MHSA(dim, heads, rope_base, rope_kind="1d")  # RoPE on (global CLS + frame CLS)

        self.n_mlp = nn.RMSNorm(dim, eps=eps)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(
        self,
        gcls: torch.Tensor,                   # (B, 1, D)
        frames: torch.Tensor,                 # (B, T, 1+Nvis, D)  (NO padding)
        hw: tuple[int, int],
        patch_pos_idx: Optional[torch.Tensor] = None,  # (B, T, Nvis)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, Lf, D = frames.shape
        n_prefix = 1
        Nvis = Lf - 1

        # (1) spatial attn per frame: (CLS + visible patches)
        x = frames.reshape(B * T, Lf, D)
        pos = patch_pos_idx.reshape(B * T, Nvis) if (patch_pos_idx is not None and Nvis > 0) else None
        x = x + self.spa(self.n_spa(x), rope_shape=hw, rope_pos_idx=pos, n_prefix=n_prefix)
        frames = x.reshape(B, T, Lf, D)

        # (2) temporal attn over CLS tokens (+ global CLS)
        fcls = frames[:, :, 0, :]                     # (B, T, D)
        tmp_in = torch.cat((gcls, fcls), dim=1)       # (B, 1+T, D)
        tmp_out = tmp_in + self.tmp(self.n_tmp(tmp_in), rope_shape=T, n_prefix=1)
        gcls = tmp_out[:, :1, :]
        frames[:, :, 0, :] = tmp_out[:, 1:, :]

        # (3) MLP over all tokens (global + all frame tokens)
        all_tok = torch.cat((gcls, frames.reshape(B, T * Lf, D)), dim=1)
        all_tok = all_tok + self.mlp(self.n_mlp(all_tok))
        gcls = all_tok[:, :1, :]
        frames = all_tok[:, 1:, :].reshape(B, T, Lf, D)
        return gcls, frames


# ----------------------------- encoder -----------------------------

@dataclass
class VideoViTCfg:
    in_chans: int = 3
    patch: int = 16
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    eps: float = 1e-6


class VideoViTEncoder(nn.Module):
    """
    MAE-style encoder where masked patch tokens are DROPPED (not padded), so SDPA sees no attn masks.

    keep_idx:
      - required for MAE-style masking
      - shape (B, T, Nvis) with values in [0, N-1]
      - Nvis must be constant across frames (as you stated)
    """

    def __init__(self, cfg: VideoViTCfg):
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed(cfg.patch, cfg.in_chans, cfg.dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.dim))   # per-frame CLS
        self.gcls = nn.Parameter(torch.zeros(1, 1, cfg.dim))  # global CLS
        self.blocks = nn.ModuleList([
            VideoBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.rope_base, cfg.eps)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.RMSNorm(cfg.dim, eps=cfg.eps)
        self._init()

    def _init(self):
        trunc_normal_(self.cls)
        trunc_normal_(self.gcls)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)

    @staticmethod
    def _check_keep_idx(keep_idx: torch.Tensor, B: int, T: int, N: int) -> torch.Tensor:
        if keep_idx.dim() != 3:
            raise ValueError(f"keep_idx must be (B,T,Nvis), got {tuple(keep_idx.shape)}")
        if keep_idx.size(0) != B or keep_idx.size(1) != T:
            raise ValueError(f"keep_idx must have (B,T,*)=({B},{T},*), got {tuple(keep_idx.shape)}")
        keep_idx = keep_idx.long()
        if keep_idx.numel() == 0:
            raise ValueError("keep_idx is empty")
        if keep_idx.min().item() < 0 or keep_idx.max().item() >= N:
            raise ValueError(f"keep_idx out of range: must be in [0,{N-1}]")
        return keep_idx

    def forward(
        self,
        x: torch.Tensor,                 # (B, T, C, H, W)
        keep_idx: torch.Tensor | None = None,          # (B, T, Nvis)
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)

        p, hw = self.patch(x_bt)         # (B*T, N, D)
        N, D = p.shape[1], p.shape[2]

        if keep_idx is None:
            if self.training:
                raise ValueError(
                    "keep_idx is required during training (MAE-style masking). "
                    "Pass keep_idx from your masking sampler.")
            else:
                keep_idx = torch.arange(N, device=x.device, dtype=torch.long)[None, None, :].expand(B, T, N)

        keep_idx = self._check_keep_idx(keep_idx, B, T, N)
        Nvis = keep_idx.size(2)

        # gather visible patches only: (B*T, Nvis, D)
        keep_bt = keep_idx.reshape(B * T, Nvis)
        idx = keep_bt.unsqueeze(-1).expand(-1, -1, D)
        p_vis = torch.gather(p, dim=1, index=idx)

        # IMPORTANT for autocast: cast fp32 params to activation dtype before cat
        cls = self.cls.to(dtype=p_vis.dtype, device=p_vis.device).expand(B * T, 1, -1)
        frames = torch.cat((cls, p_vis), dim=1).reshape(B, T, 1 + Nvis, D)

        gcls = self.gcls.to(dtype=frames.dtype, device=frames.device).expand(B, 1, -1)

        for blk in self.blocks:
            gcls, frames = blk(gcls, frames, hw, patch_pos_idx=keep_idx)

        gcls = self.norm(gcls).squeeze(1)    # (B, D)
        frames = self.norm(frames)           # (B, T, 1+Nvis, D)
        return gcls, frames, hw


# ----------------------------- decoder -----------------------------

@dataclass
class VideoViTDecCfg:
    dec_dim: int = 512
    dec_depth: int = 8
    dec_heads: int = 16
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    eps: float = 1e-6


class VideoViTDecoder(nn.Module):
    """
    MAE-style decoder.
    Inputs:
      - enc_tokens: (B, T, 1+Nvis, Denc)
      - keep_idx  : (B, T, Nvis) indices into full patch grid [0, N-1]
      - hw        : (h, w), so N=h*w
    Output:
      - pred_masked: (B, T, Nmask, patch_dim) predictions for masked patches only (per-frame patch order)
    """

    def __init__(self, enc_dim: int, patch: int, in_chans: int, cfg: VideoViTDecCfg):
        super().__init__()
        self.patch = patch
        self.in_chans = in_chans
        self.cfg = cfg

        self.proj_in = nn.Linear(enc_dim, cfg.dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
        self.gcls = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))

        self.blocks = nn.ModuleList([
            VideoBlock(cfg.dec_dim, cfg.dec_heads, cfg.mlp_ratio, cfg.rope_base, cfg.eps)
            for _ in range(cfg.dec_depth)
        ])
        self.norm = nn.RMSNorm(cfg.dec_dim, eps=cfg.eps)
        self.head = nn.Linear(cfg.dec_dim, patch * patch * in_chans, bias=True)

        self._init()

    def _init(self):
        trunc_normal_(self.mask_token)
        trunc_normal_(self.gcls)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)

    @staticmethod
    def _check_keep_idx(keep_idx: torch.Tensor, B: int, T: int, N: int) -> torch.Tensor:
        if keep_idx.dim() != 3:
            raise ValueError(f"keep_idx must be (B,T,Nvis), got {tuple(keep_idx.shape)}")
        if keep_idx.size(0) != B or keep_idx.size(1) != T:
            raise ValueError(f"keep_idx must have (B,T,*)=({B},{T},*), got {tuple(keep_idx.shape)}")
        keep_idx = keep_idx.long()
        if keep_idx.numel() == 0:
            raise ValueError("keep_idx is empty")
        if keep_idx.min().item() < 0 or keep_idx.max().item() >= N:
            raise ValueError(f"keep_idx out of range: must be in [0,{N-1}]")
        return keep_idx

    def forward(
        self,
        enc_tokens: torch.Tensor,          # (B, T, 1+Nvis, Denc)
        keep_idx: torch.Tensor,            # (B, T, Nvis)
        hw: Tuple[int, int],
    ) -> torch.Tensor:
        B, T, Lvis, Denc = enc_tokens.shape
        h, w = hw
        N = h * w
        Nvis = Lvis - 1
        if Nvis <= 0:
            raise ValueError("enc_tokens must include frame CLS + >=1 patch token")

        keep_idx = self._check_keep_idx(keep_idx, B, T, N)
        Nmask = N - Nvis
        if Nmask <= 0:
            raise ValueError("No masked tokens (Nmask<=0); increase mask ratio.")

        x = self.proj_in(enc_tokens)     # (B, T, 1+Nvis, Ddec)
        work_dtype = x.dtype
        work_dev = x.device

        fcls = x[:, :, :1, :]            # (B, T, 1, Ddec)
        vis = x[:, :, 1:, :]             # (B, T, Nvis, Ddec)
        Ddec = vis.size(-1)

        # IMPORTANT for autocast: cast fp32 params to activation dtype before use
        mask_tok = self.mask_token.to(dtype=work_dtype, device=work_dev)
        g = self.gcls.to(dtype=work_dtype, device=work_dev).expand(B, 1, -1)

        # scatter visible into full grid of mask tokens
        full = mask_tok.expand(B * T, N, Ddec).clone()  # now already correct dtype/device
        vis_bt = vis.reshape(B * T, Nvis, Ddec)
        keep_bt = keep_idx.reshape(B * T, Nvis)
        full.scatter_(1, keep_bt.unsqueeze(-1).expand(-1, -1, Ddec), vis_bt)

        patch_pos = torch.arange(N, device=work_dev).view(1, 1, N).expand(B, T, N)
        frames = torch.cat((fcls, full.view(B, T, N, Ddec)), dim=2)  # (B, T, 1+N, Ddec)

        for blk in self.blocks:
            g, frames = blk(g, frames, hw, patch_pos_idx=patch_pos)

        pred = self.head(self.norm(frames)[:, :, 1:, :])  # (B, T, N, patch_dim)

        # gather masked positions
        keep_bool = torch.zeros(B * T, N, device=work_dev, dtype=torch.bool)
        keep_bool.scatter_(1, keep_bt, True)
        mask_idx = (~keep_bool).nonzero(as_tuple=False)[:, 1].view(B * T, Nmask)  # stable order by index

        pred_bt = pred.reshape(B * T, N, -1)
        masked = torch.gather(pred_bt, 1, mask_idx.unsqueeze(-1).expand(-1, -1, pred_bt.size(-1)))
        return masked.view(B, T, Nmask, -1)


# ----------------------------- quick sanity -----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = VideoViTEncoder(VideoViTCfg(dim=384, depth=8, heads=6, patch=8)).to(device)
    vid = torch.randn(2, 8, 3, 128, 128, device=device)  # (B, T, C, H, W)

    N = (128 // 8) * (128 // 8)
    Nvis = N // 2
    keep_idx = torch.stack(
        [torch.stack([torch.randperm(N)[:Nvis] for _ in range(8)], dim=0) for _ in range(2)],
        dim=0,
    ).to(device)  # (B,T,Nvis)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        gcls, tok, hw = enc(vid, keep_idx=keep_idx)
        print("enc:", gcls.shape, tok.shape, hw)  # (B,D), (B,T,1+Nvis,D)
        print(f"enc params: {sum(p.numel() for p in enc.parameters())/1e6:.2f}M")

        dec = VideoViTDecoder(enc_dim=384, patch=8, in_chans=3, cfg=VideoViTDecCfg(dec_dim=256, dec_depth=2, dec_heads=8)).to(device)
        masked = dec(tok, keep_idx=keep_idx, hw=hw)
        print("dec:", masked.shape)  # (B,T,Nmask,patch_dim)
        print(f"dec params: {sum(p.numel() for p in dec.parameters())/1e6:.2f}M")
