from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Timestamp RoPE (1D)  (timestamps always (B,L))
# -------------------------

def trunc_normal_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return t.normal_(0.0, std).clamp_(-2 * std, 2 * std)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, L, Dh)
    cos: (B, L, Dh)
    sin: (B, L, Dh)
    """
    cos = cos[:, None, :, :]  # (B,1,L,Dh)
    sin = sin[:, None, :, :]  # (B,1,L,Dh)

    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[..., 0::2], sin[..., 0::2]
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    return torch.stack((y1, y2), dim=-1).flatten(-2)


def rope1d_cos_sin_from_pos(d: int, pos: torch.Tensor, base: float, dtype: torch.dtype):
    """
    d: head_dim (even)
    pos: (B, L) timestamps (float/int, can be irregular)
    returns:
      cos: (B, L, d)
      sin: (B, L, d)
    """
    if d % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got d={d}")
    if pos.dim() != 2:
        raise ValueError(f"timestamps must be (B,L), got {tuple(pos.shape)}")

    half = d // 2
    pos_f = pos.to(torch.float32)  # (B,L)

    inv = 1.0 / (base ** (torch.arange(half, device=pos.device, dtype=torch.float32) / half))  # (half,)

    ang = pos_f[:, :, None] * inv[None, None, :]  # (B,L,half)
    cos = ang.cos().repeat_interleave(2, dim=-1)  # (B,L,d)
    sin = ang.sin().repeat_interleave(2, dim=-1)  # (B,L,d)
    return cos.to(dtype), sin.to(dtype)


# -------------------------
# Transformer pieces
# -------------------------

class TsRoPEAttention(nn.Module):
    """
    Standard MHSA over a 1D sequence, positions come from timestamps via RoPE.
    Assumptions:
      - timestamps is always (B, L)
      - always non-causal
    """
    def __init__(self, dim: int, heads: int, rope_base: float = 10000.0, attn_dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim must be divisible by heads: dim={dim}, heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")

        self.scale = self.head_dim ** -0.5
        self.rope_base = float(rope_base)
        self.attn_dropout = float(attn_dropout)

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,          # (B, L, D)
        *,
        timestamps: torch.Tensor, # (B, L)
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        if timestamps.shape != (B, L):
            raise ValueError(f"timestamps must be (B,L)=({B},{L}), got {tuple(timestamps.shape)}")

        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, Dh)

        cos, sin = rope1d_cos_sin_from_pos(self.head_dim, timestamps, self.rope_base, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q * self.scale,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )  # (B, H, L, Dh)

        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, activation: type[nn.Module] = nn.GELU):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hid, bias=True)
        self.fc2 = nn.Linear(hid, dim, bias=True)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TsTransformerBlock(nn.Module):
    """
    Pre-LN transformer block:
      x = x + Attn(LN(x), timestamps)
      x = x + MLP(LN(x))
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0,
        eps: float = 1e-6,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = TsRoPEAttention(dim, heads, rope_base=rope_base, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, ratio=mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,          # (B, L, D)
        *,
        timestamps: torch.Tensor, # (B, L)
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), timestamps=timestamps, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------
# Full model
# -------------------------

@dataclass
class TsTransformerCfg:
    dim: int = 512
    depth: int = 4
    heads: int = 8
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    eps: float = 1e-6
    attn_dropout: float = 0.0


class TemporalTransformer1D(nn.Module):
    """
    1D sequence temporal transformer using timestamp RoPE.

    Inputs:
      x:          (B, L, D)
      timestamps: (B, L) timestamps for ALL tokens (always batched)

    Output:
      (B, L, D)
    """
    def __init__(self, cfg: TsTransformerCfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            [
                TsTransformerBlock(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    mlp_ratio=cfg.mlp_ratio,
                    rope_base=cfg.rope_base,
                    eps=cfg.eps,
                    attn_dropout=cfg.attn_dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,          # (B, L, D)
        *,
        timestamps: torch.Tensor, # (B, L)
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, timestamps=timestamps, attn_mask=attn_mask)
        return self.norm(x)


# -------------------------
# Tiny usage example
# -------------------------
if __name__ == "__main__":
    B, L, D = 2, 9, 256
    cfg = TsTransformerCfg(dim=D, depth=4, heads=8)
    model = TemporalTransformer1D(cfg)

    x = torch.randn(B, L, D)
    ts = torch.linspace(0, 1, steps=L).unsqueeze(0).expand(B, L)  # (B, L)

    y = model(x, timestamps=ts)
    print(y.shape)  # (B, L, D)