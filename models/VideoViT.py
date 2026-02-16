"""
Single-file VideoViT encoder + cross-attention decoder (NO global CLS, NO decoder temporal attn).

Encoder:
- Per-frame tokens = [frame_cls] + visible patch tokens
- Spatial self-attn per frame with 2D RoPE (prefix=frame_cls)
- Temporal self-attn ONLY over frame_cls with 1D timestamp RoPE
- MLP over all frame tokens

Decoder:
- NO decoder CLS token
- NO temporal attention
- Reconstruct full patch grid per frame using mask_token + scattered visible tokens
- Spatial CROSS-attn per frame:
    queries  = decoder patch grid tokens (N tokens)
    memory   = encoder [frame_cls] + encoder visible patch tokens (1+Nvis tokens)
  2D RoPE applied to query patches and memory patch keys (frame_cls is prefix, no RoPE)
- MLP
- Head predicts patch pixels/features
- Returns masked predictions only: (B, T, Nmask, dec_out)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------

def trunc_normal_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return t.normal_(0.0, std).clamp_(-2 * std, 2 * std)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, L, Dh)
    cos/sin: (L, Dh) or (B, L, Dh)
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


def rope1d_cos_sin_from_pos(d: int, pos: torch.Tensor, base: float, dtype: torch.dtype):
    assert d % 2 == 0
    half = d // 2

    pos_f = pos.to(torch.float32)
    inv = 1.0 / (base ** (torch.arange(half, device=pos.device, dtype=torch.float32) / half))

    if pos_f.dim() == 1:
        ang = pos_f[:, None] * inv[None, :]
        cos = ang.cos().repeat_interleave(2, dim=-1)
        sin = ang.sin().repeat_interleave(2, dim=-1)
        return cos.to(dtype), sin.to(dtype)

    if pos_f.dim() == 2:
        ang = pos_f[:, :, None] * inv[None, None, :]
        cos = ang.cos().repeat_interleave(2, dim=-1)
        sin = ang.sin().repeat_interleave(2, dim=-1)
        return cos.to(dtype), sin.to(dtype)

    raise ValueError(f"pos must be (L,) or (B,L), got {tuple(pos.shape)}")


def rope2d_cos_sin(d: int, h: int, w: int, base: float, device, dtype):
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


# -------------------------
# Core blocks
# -------------------------

class PatchEmbed(nn.Module):
    def __init__(self, patch: int, in_ch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B, C, H, W)
        x = self.proj(x)                        # (B, D, H', W')
        hw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)        # (B, N, D)
        return x, hw


class MHSA(nn.Module):
    """Self-attention with optional RoPE."""
    def __init__(self, dim: int, heads: int, rope_base: float, rope_kind: str):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.base = rope_base
        self.kind = rope_kind  # "none" | "1d_ts" | "2d"

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.register_buffer("_cos2d", None, persistent=False)
        self.register_buffer("_sin2d", None, persistent=False)
        self._shape2d = None

    def _ensure_rope2d(self, hw: tuple[int, int], device, dtype):
        if (
            self._cos2d is not None
            and self._shape2d == hw
            and self._cos2d.device == device
            and self._cos2d.dtype == dtype
        ):
            return
        self._cos2d, self._sin2d = rope2d_cos_sin(self.d, hw[0], hw[1], self.base, device, dtype)
        self._shape2d = hw

    def forward(
        self,
        x: torch.Tensor,  # (B, L, D)
        *,
        rope_shape: Optional[tuple[int, int]] = None,
        rope_pos_idx: Optional[torch.Tensor] = None,  # (B, L-n_prefix) or (L-n_prefix,)
        rope_pos: Optional[torch.Tensor] = None,      # (B, L-n_prefix) or (L-n_prefix,)
        n_prefix: int = 0,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, Dh)

        if self.kind != "none" and n_prefix < L:
            if self.kind == "1d_ts":
                if rope_pos is None:
                    raise ValueError("rope_pos (timestamps) is required for 1d_ts RoPE")
                cos, sin = rope1d_cos_sin_from_pos(self.d, rope_pos, self.base, x.dtype)

            elif self.kind == "2d":
                if rope_shape is None:
                    raise ValueError("rope_shape is required for 2d RoPE")
                self._ensure_rope2d(rope_shape, x.device, x.dtype)
                cos_full, sin_full = self._cos2d, self._sin2d

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
            else:
                raise ValueError(f"Unknown rope_kind: {self.kind}")

            q = torch.cat((q[:, :, :n_prefix], apply_rope(q[:, :, n_prefix:], cos, sin)), dim=2)
            k = torch.cat((k[:, :, :n_prefix], apply_rope(k[:, :, n_prefix:], cos, sin)), dim=2)

        y = F.scaled_dot_product_attention(
            q * self.scale,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class CrossAttn2d(nn.Module):
    """
    Cross-attention (queries from x, keys/values from mem) with optional 2D RoPE.

    Intended for decoder:
      x   : (BT, Nq, D)
      mem : (BT, 1+Nvis, D) where mem[:,0] is encoder frame_cls (prefix, no RoPE)
      q_pos_idx : (BT, Nq) patch positions in [0, N-1] (usually 0..N-1)
      k_pos_idx : (BT, Nvis) positions for mem patch tokens (keep_idx)
    """
    def __init__(self, dim: int, heads: int, rope_base: float):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.base = rope_base

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.register_buffer("_cos2d", None, persistent=False)
        self.register_buffer("_sin2d", None, persistent=False)
        self._shape2d = None

    def _ensure_rope2d(self, hw: tuple[int, int], device, dtype):
        if (
            self._cos2d is not None
            and self._shape2d == hw
            and self._cos2d.device == device
            and self._cos2d.dtype == dtype
        ):
            return
        self._cos2d, self._sin2d = rope2d_cos_sin(self.d, hw[0], hw[1], self.base, device, dtype)
        self._shape2d = hw

    def forward(
        self,
        x: torch.Tensor,                # (BT, Nq, D)
        mem: torch.Tensor,              # (BT, 1+Nvis, D)  mem[:,0] prefix
        *,
        hw: tuple[int, int],
        q_pos_idx: torch.Tensor,        # (BT, Nq)
        k_pos_idx: torch.Tensor,        # (BT, Nvis)
    ) -> torch.Tensor:
        BT, Nq, D = x.shape
        BTm, Lm, Dm = mem.shape
        if BTm != BT or Dm != D:
            raise ValueError(f"mem must be (BT,*,D)=({BT},*,{D}), got {tuple(mem.shape)}")
        if Lm < 1:
            raise ValueError("mem must include at least the encoder frame_cls token")
        if q_pos_idx.shape != (BT, Nq):
            raise ValueError(f"q_pos_idx must be (BT,Nq)=({BT},{Nq}), got {tuple(q_pos_idx.shape)}")
        if k_pos_idx.dim() != 2 or k_pos_idx.size(0) != BT:
            raise ValueError(f"k_pos_idx must be (BT,Nvis), got {tuple(k_pos_idx.shape)}")
        Nvis = k_pos_idx.size(1)
        if Lm != 1 + Nvis:
            raise ValueError(f"mem length must be 1+Nvis={1+Nvis}, got {Lm}")

        # Projections
        q = self.q_proj(x).reshape(BT, Nq, self.h, self.d).transpose(1, 2)          # (BT,H,Nq,Dh)
        kv = self.kv_proj(mem).reshape(BT, Lm, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                         # (BT,H,Lm,Dh)

        # 2D RoPE on query patches and memory patch keys (not on prefix mem[:,0])
        self._ensure_rope2d(hw, x.device, x.dtype)
        cos_full, sin_full = self._cos2d, self._sin2d                               # (N,Dh)

        # Query rope (all query tokens are patches)
        q_flat = q_pos_idx.reshape(-1)
        cos_q = cos_full.index_select(0, q_flat).view(BT, Nq, self.d)
        sin_q = sin_full.index_select(0, q_flat).view(BT, Nq, self.d)
        q = apply_rope(q, cos_q, sin_q)

        # Key rope (only patch part, skip prefix at index 0)
        k_patch = k[:, :, 1:, :]                                                    # (BT,H,Nvis,Dh)
        k_flat = k_pos_idx.reshape(-1)
        cos_k = cos_full.index_select(0, k_flat).view(BT, Nvis, self.d)
        sin_k = sin_full.index_select(0, k_flat).view(BT, Nvis, self.d)
        k_patch = apply_rope(k_patch, cos_k, sin_k)
        k = torch.cat([k[:, :, :1, :], k_patch], dim=2)

        y = F.scaled_dot_product_attention(
            q * self.scale,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )  # (BT,H,Nq,Dh)
        return self.proj(y.transpose(1, 2).reshape(BT, Nq, D))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float, activation: type[nn.Module] = nn.GELU):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hid, bias=True)
        self.fc2 = nn.Linear(hid, dim, bias=True)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


# -------------------------
# Encoder
# -------------------------

class EncoderBlock(nn.Module):
    """
    Encoder block:
      - Spatial self-attn per frame (2D RoPE, prefix=frame_cls)
      - Temporal self-attn over frame_cls only (1D timestamp RoPE)
      - MLP over all tokens
    """
    def __init__(self, dim: int, heads: int, mlp_ratio: float, rope_base: float, eps: float):
        super().__init__()
        self.n_spa = nn.LayerNorm(dim, eps=eps)
        self.spa = MHSA(dim, heads, rope_base, rope_kind="2d")

        self.n_tmp = nn.LayerNorm(dim, eps=eps)
        self.tmp = MHSA(dim, heads, rope_base, rope_kind="1d_ts")
        self.register_buffer("t_scale", torch.tensor(32.0))

        self.n_mlp = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, mlp_ratio)

    def spatial_attn(
        self,
        frames: torch.Tensor,                    # (B,T,1+Nvis,D)
        hw: tuple[int, int],
        patch_pos_idx: Optional[torch.Tensor],   # (B,T,Nvis) or None
    ) -> torch.Tensor:
        B, T, Lf, D = frames.shape
        x = frames.reshape(B * T, Lf, D)
        pos = patch_pos_idx.reshape(B * T, Lf - 1) if patch_pos_idx is not None else None
        x = x + self.spa(self.n_spa(x), rope_shape=hw, rope_pos_idx=pos, n_prefix=1)
        return x.reshape(B, T, Lf, D)

    def temporal_attn(
        self,
        frames: torch.Tensor,               # (B,T,1+Nvis,D)
        timestamps: torch.Tensor,           # (B,T)
    ) -> torch.Tensor:
        fcls = frames[:, :, 0, :]           # (B,T,D)
        tpos = timestamps.to(device=fcls.device) * self.t_scale
        fcls = fcls + self.tmp(self.n_tmp(fcls), rope_pos=tpos, n_prefix=0)
        frames = frames.clone()
        frames[:, :, 0, :] = fcls
        return frames

    def mlp_block(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, Lf, D = frames.shape
        x = frames.reshape(B, T * Lf, D)
        x = x + self.mlp(self.n_mlp(x))
        return x.reshape(B, T, Lf, D)

    def forward(
        self,
        frames: torch.Tensor,                  # (B,T,1+Nvis,D)
        hw: tuple[int, int],
        patch_pos_idx: Optional[torch.Tensor], # (B,T,Nvis)
        timestamps: torch.Tensor,              # (B,T)
    ) -> torch.Tensor:
        frames = self.spatial_attn(frames, hw, patch_pos_idx)
        frames = self.temporal_attn(frames, timestamps)
        frames = self.mlp_block(frames)
        return frames


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
    Returns:
      frame_cls:    (B,T,D)
      frame_tokens: (B,T,Nvis,D)  (visible patch tokens only)
      hw:           (H',W')
    """
    def __init__(self, cfg: VideoViTCfg):
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed(cfg.patch, cfg.in_chans, cfg.dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.dim))

        self.blocks = nn.ModuleList(
            [EncoderBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.rope_base, cfg.eps) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self._init()

    def _init(self):
        trunc_normal_(self.cls)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _check_keep_idx(keep_idx: torch.Tensor, B: int, T: int, N: int) -> torch.Tensor:
        if keep_idx.dim() != 3:
            raise ValueError(f"keep_idx must be (B,T,Nvis), got {tuple(keep_idx.shape)}")
        if keep_idx.size(0) != B or keep_idx.size(1) != T:
            raise ValueError(f"keep_idx must have (B,T,*)=({B},{T},*), got {tuple(keep_idx.shape)}")
        keep_idx = keep_idx.long()
        if keep_idx.numel() == 0:
            raise ValueError("keep_idx is empty")
        return keep_idx

    def forward(
        self,
        x: torch.Tensor,             # (B,T,C,H,W)
        keep_idx: torch.Tensor,      # (B,T,Nvis)
        timestamps: torch.Tensor,    # (B,T)
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)

        p, hw = self.patch(x_bt)     # (B*T,N,D)
        N, D = p.shape[1], p.shape[2]

        keep_idx = self._check_keep_idx(keep_idx, B, T, N)
        Nvis = keep_idx.size(2)

        keep_bt = keep_idx.reshape(B * T, Nvis)
        p_vis = torch.gather(p, dim=1, index=keep_bt.unsqueeze(-1).expand(-1, -1, D))

        cls = self.cls.to(dtype=p_vis.dtype, device=p_vis.device).expand(B * T, 1, -1)
        frames = torch.cat((cls, p_vis), dim=1).reshape(B, T, 1 + Nvis, D)

        for blk in self.blocks:
            frames = blk(frames, hw, patch_pos_idx=keep_idx, timestamps=timestamps)

        frames = self.norm(frames)
        frame_cls = frames[:, :, 0, :]      # (B,T,D)
        frame_tokens = frames[:, :, 1:, :]  # (B,T,Nvis,D)
        return frame_cls, frame_tokens, hw


# -------------------------
# Decoder (spatial cross-attn only)
# -------------------------

class DecCrossBlock(nn.Module):
    """
    Decoder block:
      - Spatial cross-attn (queries = full grid patch tokens; memory = encoder [cls]+vis tokens)
      - MLP
    No temporal attention. No decoder cls.
    """
    def __init__(self, dim: int, heads: int, mlp_ratio: float, rope_base: float, eps: float):
        super().__init__()
        self.n_q = nn.LayerNorm(dim, eps=eps)
        self.n_m = nn.LayerNorm(dim, eps=eps)
        self.xattn = CrossAttn2d(dim, heads, rope_base)

        self.n_mlp = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,              # (BT,N,Ddec) queries (full grid)
        mem: torch.Tensor,            # (BT,1+Nvis,Ddec) memory (enc cls + enc visible)
        *,
        hw: tuple[int, int],
        q_pos_idx: torch.Tensor,      # (BT,N) 0..N-1
        k_pos_idx: torch.Tensor,      # (BT,Nvis) keep indices
    ) -> torch.Tensor:
        x = x + self.xattn(self.n_q(x), self.n_m(mem), hw=hw, q_pos_idx=q_pos_idx, k_pos_idx=k_pos_idx)
        x = x + self.mlp(self.n_mlp(x))
        return x


@dataclass
class VideoViTDecCfg:
    dec_dim: int = 512
    dec_out: int = 192          # e.g., patch*patch*in_chans for pixel MAE
    dec_depth: int = 8
    dec_heads: int = 16
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    eps: float = 1e-6


class VideoViTDecoder(nn.Module):
    """
    Inputs:
      frame_cls:    (B,T,Denc)
      frame_tokens: (B,T,Nvis,Denc)
      keep_idx:     (B,T,Nvis)   patch positions for frame_tokens
      hw:           (h,w)        patch grid
    Returns:
      masked preds: (B,T,Nmask,dec_out)
    """
    def __init__(self, enc_dim: int, patch: int, in_chans: int, cfg: VideoViTDecCfg):
        super().__init__()
        self.patch = patch
        self.in_chans = in_chans
        self.cfg = cfg

        self.proj_in = nn.Linear(enc_dim, cfg.dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))

        self.blocks = nn.ModuleList(
            [DecCrossBlock(cfg.dec_dim, cfg.dec_heads, cfg.mlp_ratio, cfg.rope_base, cfg.eps)
             for _ in range(cfg.dec_depth)]
        )
        self.norm = nn.LayerNorm(cfg.dec_dim, eps=cfg.eps)
        self.head = nn.Linear(cfg.dec_dim, cfg.dec_out, bias=True)

        self._init()

    def _init(self):
        trunc_normal_(self.mask_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _check_keep_idx(keep_idx: torch.Tensor, B: int, T: int, N: int) -> torch.Tensor:
        if keep_idx.dim() != 3:
            raise ValueError(f"keep_idx must be (B,T,Nvis), got {tuple(keep_idx.shape)}")
        if keep_idx.size(0) != B or keep_idx.size(1) != T:
            raise ValueError(f"keep_idx must have (B,T,*)=({B},{T},*), got {tuple(keep_idx.shape)}")
        keep_idx = keep_idx.long()
        if keep_idx.numel() == 0:
            raise ValueError("keep_idx is empty")
        return keep_idx

    def forward(
        self,
        frame_cls: torch.Tensor,           # (B,T,Denc)
        frame_tokens: torch.Tensor,        # (B,T,Nvis,Denc)
        keep_idx: torch.Tensor,            # (B,T,Nvis)
        hw: Tuple[int, int],
    ) -> torch.Tensor:
        B, T, Nvis, Denc = frame_tokens.shape
        h, w = hw
        N = h * w

        keep_idx = self._check_keep_idx(keep_idx, B, T, N)

        Nmask = N - Nvis
        if Nmask <= 0:
            raise ValueError("No masked tokens (Nmask<=0)")

        # Project encoder outputs to decoder dim
        cls_d = self.proj_in(frame_cls)           # (B,T,Ddec)
        tok_d = self.proj_in(frame_tokens)        # (B,T,Nvis,Ddec)
        Ddec = tok_d.size(-1)

        # Build per-frame memory: [enc_cls] + enc_visible_tokens
        mem = torch.cat([cls_d.unsqueeze(2), tok_d], dim=2)   # (B,T,1+Nvis,Ddec)
        mem_bt = mem.reshape(B * T, 1 + Nvis, Ddec)           # (BT,1+Nvis,Ddec)

        # Build query grid (no decoder cls): mask everywhere then scatter visible tokens
        BT = B * T
        keep_bt = keep_idx.reshape(BT, Nvis)

        x = self.mask_token.to(dtype=mem_bt.dtype, device=mem_bt.device).expand(BT, N, -1).clone()
        vis_bt = tok_d.reshape(BT, Nvis, Ddec)
        x.scatter_(1, keep_bt.unsqueeze(-1).expand(-1, -1, Ddec), vis_bt)            # (BT,N,Ddec)

        # Pos indices
        q_pos = torch.arange(N, device=mem_bt.device).view(1, N).expand(BT, N)       # (BT,N)

        # Cross-attend (spatial only) + MLP
        for blk in self.blocks:
            x = blk(x, mem_bt, hw=hw, q_pos_idx=q_pos, k_pos_idx=keep_bt)

        pred = self.head(self.norm(x))                                                # (BT,N,dec_out)

        # Return masked only (same ascending mask order as before)
        keep_bool = torch.zeros(BT, N, device=pred.device, dtype=torch.bool)
        keep_bool.scatter_(1, keep_bt, True)
        mask_idx = (~keep_bool).nonzero(as_tuple=False)[:, 1].view(BT, Nmask)        # (BT,Nmask)

        masked = torch.gather(pred, 1, mask_idx.unsqueeze(-1).expand(BT, Nmask, pred.size(-1)))
        return masked.view(B, T, Nmask, -1)