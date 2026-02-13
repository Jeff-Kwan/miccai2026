from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return t.normal_(0.0, std).clamp_(-2 * std, 2 * std)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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


class PatchEmbed(nn.Module):
    def __init__(self, patch: int, in_ch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
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
        self.kind = rope_kind  # "none" | "1d_ts" | "2d"

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.register_buffer("_cos2d", None, persistent=False)
        self.register_buffer("_sin2d", None, persistent=False)
        self._shape2d = None

    def _ensure_rope2d(self, hw, device, dtype):
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
        rope_shape=None,
        rope_pos_idx: Optional[torch.Tensor] = None,  # (B, L-n_prefix) or (L-n_prefix,)
        rope_pos: Optional[torch.Tensor] = None,      # (B, L-n_prefix) or (L-n_prefix,)
        n_prefix: int = 0,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.kind != "none" and n_prefix < L:
            if self.kind == "1d_ts":
                if rope_pos is None:
                    raise ValueError("rope_pos (timestamps) is required for 1d_ts RoPE")
                if rope_pos.dim() == 1:
                    if rope_pos.numel() != (L - n_prefix):
                        raise ValueError(f"rope_pos must have length {L-n_prefix}, got {rope_pos.numel()}")
                    cos, sin = rope1d_cos_sin_from_pos(self.d, rope_pos, self.base, x.dtype)
                elif rope_pos.dim() == 2:
                    if rope_pos.size(0) != B or rope_pos.size(1) != (L - n_prefix):
                        raise ValueError(
                            f"rope_pos must be (B,{L-n_prefix})=({B},{L-n_prefix}), got {tuple(rope_pos.shape)}"
                        )
                    cos, sin = rope1d_cos_sin_from_pos(self.d, rope_pos, self.base, x.dtype)
                else:
                    raise ValueError(f"rope_pos must be (L,) or (B,L), got {tuple(rope_pos.shape)}")

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


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float, activation: nn.Module = nn.GELU):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hid, bias=True)
        self.fc2 = nn.Linear(hid, dim, bias=True)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class VideoBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, 
                 rope_base: float, eps: float, config: str):
        super().__init__()
        self.config = config
        self.n_spa = nn.LayerNorm(dim, eps=eps)
        self.spa = MHSA(dim, heads, rope_base, rope_kind="2d")

        self.n_tmp = nn.LayerNorm(dim, eps=eps)
        self.tmp = MHSA(dim, heads, rope_base, rope_kind="1d_ts")
        self.register_buffer('t_scale', torch.tensor(32.0))

        self.n_mlp = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, mlp_ratio)

    def spatial_attn(self, frames: torch.Tensor, hw: tuple[int, int], patch_pos_idx: Optional[torch.Tensor]):
        B, T, Lf, D = frames.shape
        x = frames.reshape(B * T, Lf, D)
        pos = patch_pos_idx.reshape(B * T, Lf - 1) if patch_pos_idx is not None else None
        x = x + self.spa(self.n_spa(x), rope_shape=hw, rope_pos_idx=pos, n_prefix=1)
        return x.reshape(B, T, Lf, D)

    def temporal_attn(self, gcls: torch.Tensor, frames: torch.Tensor, timestamps: torch.Tensor):
        fcls = frames[:, :, 0, :]
        tmp_in = torch.cat((gcls, fcls), dim=1)
        tpos = timestamps.to(device=tmp_in.device) * self.t_scale
        tmp_out = tmp_in + self.tmp(self.n_tmp(tmp_in), rope_pos=tpos, n_prefix=1)
        gcls = tmp_out[:, :1, :]
        frames[:, :, 0, :] = tmp_out[:, 1:, :]
        return gcls, frames

    def mlp_block(self, gcls: torch.Tensor, frames: torch.Tensor):
        B, T, Lf, D = frames.shape
        all_tok = torch.cat((gcls, frames.reshape(B, T * Lf, D)), dim=1)
        all_tok = all_tok + self.mlp(self.n_mlp(all_tok))
        gcls = all_tok[:, :1, :]
        frames = all_tok[:, 1:, :].reshape(B, T, Lf, D)
        return gcls, frames

    def forward(
        self,
        gcls: torch.Tensor,                   # (B, 1, D)
        frames: torch.Tensor,                 # (B, T, 1+Nvis, D)
        hw: tuple[int, int],
        patch_pos_idx: Optional[torch.Tensor],# (B, T, Nvis)
        timestamps: torch.Tensor,             # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config == 'enc':
            # Encoder is spatial - temporal - mlp
            frames = self.spatial_attn(frames, hw, patch_pos_idx)
            gcls, frames = self.temporal_attn(gcls, frames, timestamps)
            gcls, frames = self.mlp_block(gcls, frames)
        elif self.config == 'dec':
            # Decoder is temporal - spatial - mlp
            gcls, frames = self.temporal_attn(gcls, frames, timestamps)
            frames = self.spatial_attn(frames, hw, patch_pos_idx)
            gcls, frames = self.mlp_block(gcls, frames)
        else:
            raise ValueError(f"Unknown video transformer block config: {self.config}")
        return gcls, frames


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
    def __init__(self, cfg: VideoViTCfg):
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed(cfg.patch, cfg.in_chans, cfg.dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.gcls = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.blocks = nn.ModuleList([
            VideoBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.rope_base, cfg.eps, config='enc')
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.dim, eps=cfg.eps)
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
        if keep_idx.min() < 0 or keep_idx.max() >= N:
            raise ValueError(f"keep_idx out of range: must be in [0,{N-1}]")
        return keep_idx

    def forward(
        self,
        x: torch.Tensor,             # (B, T, C, H, W)
        keep_idx: torch.Tensor,      # (B, T, Nvis)
        timestamps: torch.Tensor,    # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)

        p, hw = self.patch(x_bt)
        N, D = p.shape[1], p.shape[2]

        keep_idx = self._check_keep_idx(keep_idx, B, T, N)
        Nvis = keep_idx.size(2)

        keep_bt = keep_idx.reshape(B * T, Nvis)
        idx = keep_bt.unsqueeze(-1).expand(-1, -1, D)
        p_vis = torch.gather(p, dim=1, index=idx)

        cls = self.cls.to(dtype=p_vis.dtype, device=p_vis.device).expand(B * T, 1, -1)
        frames = torch.cat((cls, p_vis), dim=1).reshape(B, T, 1 + Nvis, D)

        gcls = self.gcls.to(dtype=frames.dtype, device=frames.device).expand(B, 1, -1)

        for blk in self.blocks:
            gcls, frames = blk(gcls, frames, hw, patch_pos_idx=keep_idx, timestamps=timestamps)

        gcls = self.norm(gcls).squeeze(1)
        frames = self.norm(frames)
        return gcls, frames, hw


@dataclass
class VideoViTDecCfg:
    dec_dim: int = 512
    dec_out: int = 192
    dec_depth: int = 8
    dec_heads: int = 16
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    eps: float = 1e-6


class VideoViTDecoder(nn.Module):
    def __init__(self, enc_dim: int, patch: int, in_chans: int, cfg: VideoViTDecCfg):
        super().__init__()
        self.patch = patch
        self.in_chans = in_chans
        self.cfg = cfg

        self.proj_in = nn.Linear(enc_dim, cfg.dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))

        self.blocks = nn.ModuleList([
            VideoBlock(cfg.dec_dim, cfg.dec_heads, cfg.mlp_ratio, cfg.rope_base, cfg.eps, config='dec')
            for _ in range(cfg.dec_depth)
        ])
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
        if keep_idx.min() < 0 or keep_idx.max() >= N:
            raise ValueError(f"keep_idx out of range: must be in [0,{N-1}]")
        return keep_idx

    def forward(
        self,
        gcls: torch.Tensor,                   # (B, 1, Denc)
        enc_tokens: torch.Tensor,      # (B, T, 1+Nvis, Denc)
        keep_idx: torch.Tensor,        # (B, T, Nvis)
        hw: Tuple[int, int],
        timestamps: torch.Tensor,      # (B, T)
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
            raise ValueError("No masked tokens (Nmask<=0)")

        gcls = self.proj_in(gcls).unsqueeze(1)
        x = self.proj_in(enc_tokens)
        work_dtype, work_dev = x.dtype, x.device

        fcls = x[:, :, :1, :]
        vis = x[:, :, 1:, :]
        Ddec = vis.size(-1)

        full = self.mask_token.expand(B * T, N, -1).to(dtype=work_dtype, device=work_dev).clone()
        vis_bt = vis.reshape(B * T, Nvis, Ddec)
        keep_bt = keep_idx.reshape(B * T, Nvis)
        full.scatter_(1, keep_bt.unsqueeze(-1).expand(-1, -1, Ddec), vis_bt)

        patch_pos = torch.arange(N, device=work_dev).view(1, 1, N).expand(B, T, N)
        frames = torch.cat((fcls, full.view(B, T, N, Ddec)), dim=2)

        for blk in self.blocks:
            gcls, frames = blk(gcls, frames, hw, patch_pos_idx=patch_pos, timestamps=timestamps)

        pred = self.head(self.norm(frames)[:, :, 1:, :])

        keep_bool = torch.zeros(B * T, N, device=work_dev, dtype=torch.bool)
        keep_bool.scatter_(1, keep_bt, True)
        mask_idx = (~keep_bool).nonzero(as_tuple=False)[:, 1].view(B * T, Nmask)

        pred_bt = pred.reshape(B * T, N, -1)
        masked = torch.gather(pred_bt, 1, mask_idx.unsqueeze(-1).expand(-1, -1, pred_bt.size(-1)))
        return masked.view(B, T, Nmask, -1)