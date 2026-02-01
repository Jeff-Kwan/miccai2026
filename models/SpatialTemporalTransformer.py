import math
import torch
import torch.nn as nn


def sinusoidal_embedding_1d(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros(length, 0, device=device)
    half = (dim + 1) // 2
    pos = torch.arange(length, device=device, dtype=torch.float32)
    div = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32) *
                    (-math.log(10000.0) / half))
    sin = torch.sin(pos[:, None] * div[None, :])
    cos = torch.cos(pos[:, None] * div[None, :])
    emb = torch.cat([sin, cos], dim=1)
    return emb[:, :dim]


class LatentPerceiverBlock(nn.Module):
    """
    frames_vis:  [B, T, K, C]   (K visible tokens per frame)
    frames_full: [B, T, S, C]   (S = H*W full tokens per frame, created before write-back)
    latents:     [B, T, L, C]
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_latent_in = nn.LayerNorm(dim)
        self.norm_frame_kv  = nn.LayerNorm(dim)
        self.cross_latent_from_frame = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_latent_sa = nn.LayerNorm(dim)
        self.latent_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        hidden = int(dim * mlp_ratio)
        self.latent_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        self.norm_frame_q   = nn.LayerNorm(dim)
        self.norm_latent_kv = nn.LayerNorm(dim)
        self.cross_frame_from_latent = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.frames_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, frames_vis: torch.Tensor, frames_full: torch.Tensor, latents: torch.Tensor):
        B, T, K, C = frames_vis.shape
        _, _, S, _ = frames_full.shape
        _, _, L, _ = latents.shape

        # (a) latents <- visible frame tokens (per-frame)
        q  = self.norm_latent_in(latents.reshape(B*T, L, C))
        kv = self.norm_frame_kv(frames_vis.reshape(B*T, K, C))
        upd_lat, _ = self.cross_latent_from_frame(q, kv, kv, need_weights=False)
        latents = latents + upd_lat.reshape(B, T, L, C)

        # (b) global latent self-attn over all time
        lat_flat = latents.reshape(B, T*L, C)
        lat_ln = self.norm_latent_sa(lat_flat)
        lat_sa, _ = self.latent_self_attn(lat_ln, lat_ln, lat_ln, need_weights=False)
        lat_flat = lat_flat + lat_sa

        # (c) latent MLP
        lat_flat = lat_flat + self.latent_mlp(lat_flat)
        latents = lat_flat.reshape(B, T, L, C)

        # (d) full frames <- latents (per-frame); masked slots get updated too
        qf  = self.norm_frame_q(frames_full.reshape(B*T, S, C))
        kvl = self.norm_latent_kv(latents.reshape(B*T, L, C))
        upd_frames, _ = self.cross_frame_from_latent(qf, kvl, kvl, need_weights=False)
        frames_full = frames_full + upd_frames.reshape(B, T, S, C)

        # (e) frames MLP on full frames
        frames_full = frames_full + self.frames_mlp(frames_full)

        return frames_full, latents


class SpatialTemporalLatentTransformer(nn.Module):
    """
    Input: x [B, C, T, H, W]
     behavior:
      - add pos to get tokens-with-pos
      - per-frame keep K tokens (K=int(keep_ratio*H*W))
      - latents attend to visible tokens
      - reconstruct full tokens by inserting (mask_token + pos) in masked slots
      - full tokens attend to latents (write-back)
    Output: [B, C, T, H, W]
    Optionally returns mask info for loss computation.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        num_latents: int = 8,
        mlp_ratio: float = 4.0,
        pos_hidden_mult: int = 3,
        keep_ratio: float = 0.25,
        return_mask: bool = False,
        do_masking: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_latents = num_latents
        self.keep_ratio = keep_ratio
        self.return_mask = return_mask
        self.do_masking = do_masking

        # frame positional encoding (same as before)
        self.pos_dim = pos_hidden_mult * dim
        self.pos_proj = nn.Conv3d(self.pos_dim, dim, kernel_size=1, bias=False)

        # learned latent tokens
        self.latent_base = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        # learned mask token (shared across all masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            LatentPerceiverBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def build_frame_pos(self, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        P = self.pos_dim
        Ct = P // 3
        Ch = P // 3
        Cw = P - Ct - Ch

        t = sinusoidal_embedding_1d(T, Ct, device)
        h = sinusoidal_embedding_1d(H, Ch, device)
        w = sinusoidal_embedding_1d(W, Cw, device)

        t = t[:, None, None, :].expand(T, H, W, Ct)
        h = h[None, :, None, :].expand(T, H, W, Ch)
        w = w[None, None, :, :].expand(T, H, W, Cw)

        pos = torch.cat([t, h, w], dim=-1)          # [T,H,W,P]
        pos = pos.permute(3, 0, 1, 2).unsqueeze(0)  # [1,P,T,H,W]
        return pos

    def build_latent_temporal_pos(self, T: int, device: torch.device) -> torch.Tensor:
        tpos = sinusoidal_embedding_1d(T, self.dim, device)  # [T,C]
        return tpos[None, :, None, :]                        # [1,T,1,C]

    @staticmethod
    def _random_keep_indices(B: int, T: int, S: int, K: int, device: torch.device):
        """
        Returns:
          keep_idx: [B,T,K] indices into S
          mask:     [B,T,S] 1 for masked, 0 for kept ( convention varies; adjust as you like)
        """
        # sample independent permutations per (B,T)
        noise = torch.rand(B, T, S, device=device)
        ids = torch.argsort(noise, dim=-1)          # [B,T,S]
        keep_idx = ids[:, :, :K]                   # [B,T,K]

        mask = torch.ones(B, T, S, device=device, dtype=torch.float32)
        mask.scatter_(dim=-1, index=keep_idx, value=0.0)     # 0=keep, 1=mask
        return keep_idx, mask

    @staticmethod
    def _all_keep_indices(B: int, T: int, S: int, device: torch.device):
        # keep everything: [B,T,S]
        keep_idx = torch.arange(S, device=device).view(1, 1, S).expand(B, T, S).contiguous()
        mask = torch.zeros(B, T, S, device=device, dtype=torch.float32)  # 0=kept
        return keep_idx, mask

    def forward(self, x: torch.Tensor):
        # x: [B,C,T,H,W]
        B, C, T, H, W = x.shape
        if C != self.dim:
            raise ValueError(f"Expected C={self.dim}, got {C}")
        S = H * W

        # positional encoding
        pos_raw = self.build_frame_pos(T, H, W, x.device)  # [1, pos_dim, T,H,W]
        pos = self.pos_proj(pos_raw)                       # [1, C,      T,H,W]
        x = x + pos

        # flatten to tokens: [B,T,S,C]
        frames_tokens = x.permute(0, 2, 3, 4, 1).reshape(B, T, S, C)

        # pos flat for masked insertion
        pos_flat = pos.permute(0, 2, 3, 4, 1).reshape(1, T, S, C)  # [1,T,S,C]

        if self.do_masking:
            K = max(1, int(self.keep_ratio * S))
            keep_idx, mask = self._random_keep_indices(B, T, S, K, x.device)  # [B,T,K], [B,T,S]
        else:
            K = S
            keep_idx, mask = self._all_keep_indices(B, T, S, x.device)        # [B,T,S], [B,T,S]

        keep_idx_exp = keep_idx.unsqueeze(-1).expand(B, T, K, C)

        # fixed read stream
        visible_x = frames_tokens.gather(dim=2, index=keep_idx_exp).clone()    # [B,T,K,C]

        # evolving write stream
        if self.do_masking:
            recon = self.mask_token.expand(B, T, S, C) + pos_flat.expand(B, T, S, C)
            recon = recon.scatter(dim=2, index=keep_idx_exp, src=visible_x)
        else:
            # no masked slots; just start from the full tokens
            recon = frames_tokens

        # latents [B,T,L,C]
        latents = self.latent_base.expand(B, -1, -1).unsqueeze(1).expand(B, T, -1, -1).contiguous()
        latents = latents + self.build_latent_temporal_pos(T, x.device)

        for blk in self.blocks:
            recon, latents = blk(frames_vis=visible_x, frames_full=recon, latents=latents)

        out = recon.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        if self.return_mask:
            return out, mask, keep_idx
        return out



# ---- quick sanity check ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, T, H, W = 8, 128, 64, 16, 16
    x = torch.randn(B, C, T, H, W, device=device)

    model = SpatialTemporalLatentTransformer(
        dim=C,
        depth=6,
        num_heads=4,
        num_latents=8,
        mlp_ratio=4.0,
        do_masking=True,
    ).to(device)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
        ],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        out = model(x)

    assert out.shape == x.shape, f"Expected output shape {x.shape}, got {out.shape}"
    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    print("Total trainable parameters:",
          round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2), "M")
    print("IO is size:", x.element_size() * x.nelement() / 1024 / 1024, "MB")
    print("I/O has elements:", round(out.nelement() / 1e6, 2), "M")
