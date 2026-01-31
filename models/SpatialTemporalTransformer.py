import math
import torch
import torch.nn as nn


def sinusoidal_embedding_1d(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Standard sin/cos embedding.
    Returns: [length, dim]
    """
    if dim <= 0:
        return torch.zeros(length, 0, device=device)
    half = (dim + 1) // 2  # number of sin (and cos) frequencies we generate
    pos = torch.arange(length, device=device, dtype=torch.float32)  # [L]
    div = torch.exp(
        torch.arange(0, half, device=device, dtype=torch.float32) * (-math.log(10000.0) / half)
    )  # [half]

    sin = torch.sin(pos[:, None] * div[None, :])  # [L, half]
    cos = torch.cos(pos[:, None] * div[None, :])  # [L, half]
    emb = torch.cat([sin, cos], dim=1)            # [L, 2*half]
    return emb[:, :dim]                           # crop to requested dim


class SpatialTemporalBlock(nn.Module):
    """
    x: [B, C, T, H, W]
    spatial attn over HW (per t) -> temporal attn over T (per (h,w)) -> MLP (per token)
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim

        self.norm_s = nn.LayerNorm(dim)
        self.s_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_t = nn.LayerNorm(dim)
        self.t_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_m = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        HW = H * W

        # ---- spatial attention over HW (independent per time) ----
        xs = x.permute(0, 2, 3, 4, 1).reshape(B * T, HW, C)  # [B*T, HW, C]
        xs_ln = self.norm_s(xs)
        xs_attn, _ = self.s_attn(xs_ln, xs_ln, xs_ln, need_weights=False)
        xs = xs + xs_attn
        x = xs.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # [B,C,T,H,W]

        # ---- temporal attention over T (independent per spatial position) ----
        xt = x.permute(0, 3, 4, 2, 1).reshape(B * HW, T, C)  # [B*HW, T, C]
        xt_ln = self.norm_t(xt)
        xt_attn, _ = self.t_attn(xt_ln, xt_ln, xt_ln, need_weights=False)
        xt = xt + xt_attn
        x = xt.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2).contiguous()  # [B,C,T,H,W]

        # ---- MLP per token ----
        xm = x.permute(0, 2, 3, 4, 1).reshape(B * T * HW, C)  # [B*T*HW, C]
        xm = xm + self.mlp(self.norm_m(xm))
        x = xm.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        return x


class SpatialTemporalTransformer(nn.Module):
    """
    Input:  x [B, C, T, H, W]   (already tokenized)
    Adds sinusoidal spatiotemporal positional encoding, then projects it with a 1x1x1 conv:
        pos_raw: [1, 3C, T, H, W]  -> pos: [1, C, T, H, W]
    Then applies blocks: spatial attn -> temporal attn -> MLP
    Output: x [B, C, T, H, W]
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        pos_hidden_mult: int = 3,
        need_pos: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.pos_dim = pos_hidden_mult * dim  # 3C by default

        # project positional channels (3C -> C)
        self.need_pos = need_pos
        if need_pos:
            self.pos_proj = nn.Conv3d(self.pos_dim, dim, kernel_size=1, bias=False)

        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def build_pos(self, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Build sinusoidal pos encoding with total channels = 3C (not requiring any divisibility).
        We split 3C into (Ct, Ch, Cw) that sum to 3C.
        Returns: [1, 3C, T, H, W]
        """
        P = self.pos_dim  # 3C
        # Split as evenly as possible
        Ct = P // 3
        Ch = P // 3
        Cw = P - Ct - Ch  # remainder

        t = sinusoidal_embedding_1d(T, Ct, device)  # [T, Ct]
        h = sinusoidal_embedding_1d(H, Ch, device)  # [H, Ch]
        w = sinusoidal_embedding_1d(W, Cw, device)  # [W, Cw]

        # Broadcast to [T, H, W, *]
        t = t[:, None, None, :].expand(T, H, W, Ct)
        h = h[None, :, None, :].expand(T, H, W, Ch)
        w = w[None, None, :, :].expand(T, H, W, Cw)

        pos = torch.cat([t, h, w], dim=-1)          # [T, H, W, 3C]
        pos = pos.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3C, T, H, W]
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        if C != self.dim:
            raise ValueError(f"Expected embedding dim C={self.dim}, got C={C}")

        if self.need_pos:
            pos_raw = self.build_pos(T, H, W, x.device)  # [1, 3C, T, H, W]
            pos = self.pos_proj(pos_raw)                 # [1, C,  T, H, W]
            x = x + pos

        for blk in self.blocks:
            x = blk(x)

        return x


# ---- quick sanity check ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, T, H, W = 8, 128, 64, 16, 16
    x = torch.randn(B, C, T, H, W, device=device)
    model = SpatialTemporalTransformer(dim=C, depth=6, num_heads=4).to(device)

    # Profile memory usage
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, 
                    torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        output = model(x)

    assert output.shape == x.shape, f"Expected output shape {x.shape}, got {output.shape}"
    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1048**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1048 / 1048, 'MB')
    print("I/O has elements: ", round(output.nelement() / 1e6, 2), 'M')
