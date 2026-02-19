import torch
from torch import nn, autocast
import torch.nn.functional as F

# -------------------------
# 1D Conv blocks
# -------------------------

class ResBlock1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convs(x)


class Downsample1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# -------------------------
# 1D Encoder / Decoder
# -------------------------

class SimpleConvEncoder1D(nn.Module):
    """
    Input:  x of shape [B, T, in_dim, L]
    Output: z of shape [B, T, latent]  (one vector per step)
    """

    def __init__(self, latent: int, in_dim: int = 3):
        super().__init__()
        base = latent // 2
        assert base // 64 > 4, "Latent dimension too small for 1D architecture (try reducing number of downsample stages)"
        # Channel schedule (mirrors the 2D version, just 1D ops)
        channels = [
            base // 64,
            base // 32,
            base // 16,
            base // 8,
            base // 4,
            base // 2,
            base,
            base,
        ]

        self.in_conv = nn.Conv1d(in_dim, channels[0], kernel_size=3, stride=1, padding=1)

        self.stages = nn.ModuleList()
        for i in range(1, len(channels)):
            self.stages.append(
                nn.Sequential(
                    Downsample1D(channels[i - 1], channels[i]),
                    ResBlock1D(channels[i]),
                    ResBlock1D(channels[i]),
                )
            )

        self.to_latent = nn.Conv1d(channels[-1], latent, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, L)
        B, T, C, L = x.shape
        x = x.reshape(B * T, C, L)

        h = self.in_conv(x)
        for stage in self.stages:
            h = stage(h)

        h = self.to_latent(h)        # (B*T, latent, lL)
        h = self.pool(h)             # (B*T, latent, 1)
        z = h.squeeze(-1).reshape(B, T, -1)  # (B, T, latent)
        z = self.latent_proj(z)
        return z


class SimpleConvDecoder1D(nn.Module):
    """
    Input:  z of shape [B, T, latent]
    Output: x of shape [B, T, out_dim, L]
    """

    def __init__(self, latent: int, out_dim: int = 3):
        super().__init__()
        base = latent // 2

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent, base, kernel_size=2, stride=2, padding=0),  # 1 -> 2
            Upsample1D(base,      max(1, base // 2)),   # 2 -> 4
            Upsample1D(max(1, base // 2), max(1, base // 4)),  # 4 -> 8
            Upsample1D(max(1, base // 4), max(1, base // 8)),  # 8 -> 16
            Upsample1D(max(1, base // 8), max(1, base // 16)), # 16 -> 32
            Upsample1D(max(1, base // 16), max(1, base // 32)),# 32 -> 64
            Upsample1D(max(1, base // 32), max(1, base // 64)),
            nn.Conv1d(max(1, base // 64), out_dim, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, L: int) -> torch.Tensor:
        # z: (B, T, latent)
        B, T, latent = z.shape
        z = z.reshape(B * T, latent, 1)  # (B*T, latent, 1)
        x = self.decoder(z)              # nominal (B*T, out_dim, 64)
        x = F.interpolate(x, size=L, mode="linear", align_corners=False)
        return x.reshape(B, T, -1, L)    # (B, T, out_dim, L)


# -------------------------
# SplineAutoEncoder1D
# -------------------------

class SplineAutoEncoder1D(nn.Module):
    """
    Autoencoder + cubic B-spline smoothing in latent space (1D encoder/decoder).

    Spline fit uses a second-difference penalty on control points:
        argmin_P ||A P - Z||^2 + lam ||D P||^2
    """

    def __init__(
        self,
        latent: int,
        in_dim: int = 3,
        out_dim: int | None = None,
        degree: int = 3,
        n_ctrl: int = 12,
        lam: float = 1e-4,
        eps: float = 1e-12,
    ):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim

        self.latent = latent
        self.encoder = SimpleConvEncoder1D(latent=latent, in_dim=in_dim)
        self.decoder = SimpleConvDecoder1D(latent=latent, out_dim=out_dim)

        self.degree = degree
        self.n_ctrl = n_ctrl
        self.lam = lam
        self.eps = eps

    # -------------------------
    # Encode / decode
    # -------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, L) -> (B, T, latent)
        return self.encoder(x)

    def decode(self, z: torch.Tensor, L: int) -> torch.Tensor:
        # (B, T, latent) -> (B, T, C, L)
        return self.decoder(z, L=L)

    # -------------------------
    # B-spline utilities
    # -------------------------

    @staticmethod
    def make_clamped_uniform_knots(t: torch.Tensor, degree: int, n_ctrl: int) -> torch.Tensor:
        """
        Clamped uniform knot vector per batch.
        t: (B, T)  (typically normalized already)
        returns: (B, n_ctrl + degree + 1)
        """
        B = t.shape[0]
        t0 = t.min(dim=1).values
        t1 = t.max(dim=1).values

        n_knots = n_ctrl + degree + 1
        n_int = n_knots - 2 * (degree + 1)  # number of interior knots

        if n_int > 0:
            u = torch.linspace(0, 1, steps=n_int + 2, device=t.device, dtype=t.dtype)[1:-1]
            interior = t0[:, None] + (t1 - t0)[:, None] * u[None, :]
        else:
            interior = t.new_empty(B, 0)

        left = t0[:, None].repeat(1, degree + 1)
        right = t1[:, None].repeat(1, degree + 1)
        return torch.cat([left, interior, right], dim=1)

    @staticmethod
    def bspline_basis(t: torch.Tensor, knots: torch.Tensor, degree: int, eps: float = 1e-12):
        B, T = t.shape
        K = knots.shape[1]
        n_ctrl = K - degree - 1

        t_ = t[:, :, None]
        k0 = knots[:, None, :-1]
        k1 = knots[:, None, 1:]

        # p=0 basis (half-open)
        N = ((t_ >= k0) & (t_ < k1)).to(t.dtype)  # (B, T, K-1)

        tol = 10 * torch.finfo(t.dtype).eps
        last_knot = knots[:, -1][:, None]        # (B, 1)
        is_last = t >= (last_knot - tol)         # (B, T)

        # If exactly at the end, assign to the last interval for p=0
        N = torch.where(is_last[:, :, None], torch.zeros_like(N), N)
        N[:, :, -1] = N[:, :, -1] + is_last.to(t.dtype)

        def safe_div(num, den):
            return torch.where(den.abs() > tol, num / den, torch.zeros_like(num))

        for p in range(1, degree + 1):
            denom1 = knots[:, None, p:K - 1] - knots[:, None, :K - 1 - p]
            denom2 = knots[:, None, p + 1:K] - knots[:, None, 1:K - p]

            w1 = safe_div(t_ - knots[:, None, :K - 1 - p], denom1)
            w2 = safe_div(knots[:, None, p + 1:K] - t_, denom2)

            N = w1 * N[:, :, :K - 1 - p] + w2 * N[:, :, 1:K - p]

        N = N[:, :, :n_ctrl]

        # Strong endpoint guarantee for degree-p basis too:
        N = torch.where(is_last[:, :, None], torch.zeros_like(N), N)
        N[:, :, -1] = N[:, :, -1] + is_last.to(t.dtype)

        return N

    @staticmethod
    def second_difference_gram(n_ctrl: int, device, dtype) -> torch.Tensor:
        """
        Returns DtD where D is the (n_ctrl-2) x n_ctrl second-difference matrix:
            (D p)_i = p_i - 2 p_{i+1} + p_{i+2}
        """
        if n_ctrl < 3:
            return torch.zeros((n_ctrl, n_ctrl), device=device, dtype=dtype)

        D = torch.zeros((n_ctrl - 2, n_ctrl), device=device, dtype=dtype)
        i = torch.arange(n_ctrl - 2, device=device)
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
        return D.transpose(0, 1) @ D

    # -------------------------
    # Spline fit + eval
    # -------------------------

    def spline_fit(self, z_in: torch.Tensor, t_in: torch.Tensor):
        """
        Fit spline control points for z_in over t_in.

        Returns:
        P32:   (B, n_ctrl, latent) fp32
        knots: (B, n_ctrl + degree + 1) fp32
        t0,t1: (B, 1) fp32
        """
        device = z_in.device
        with autocast("cuda", enabled=False):
            z32 = z_in.float()
            t32 = t_in.float()

            t0 = t32.min(dim=1, keepdim=True).values
            t1 = t32.max(dim=1, keepdim=True).values
            t_n = (t32 - t0) / (t1 - t0 + self.eps)

            knots = self.make_clamped_uniform_knots(t_n, self.degree, self.n_ctrl)
            A = self.bspline_basis(t_n, knots, self.degree, eps=self.eps)

            At = A.transpose(1, 2)
            AtA = At @ A
            AtZ = At @ z32

            DtD = self.second_difference_gram(self.n_ctrl, device=device, dtype=torch.float32)
            lhs = AtA + self.lam * DtD[None, :, :]

            P32 = torch.linalg.solve(lhs, AtZ)

        return P32, knots, t0, t1

    def spline_eval(
        self,
        P32: torch.Tensor,      # (B, n_ctrl, latent) fp32
        knots: torch.Tensor,    # (B, K) fp32
        t: torch.Tensor,        # (B, T)
        t0: torch.Tensor,       # (B, 1) fp32
        t1: torch.Tensor,       # (B, 1) fp32
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        with autocast("cuda", enabled=False):
            t32 = t.float()
            t_n = (t32 - t0) / (t1 - t0 + self.eps)
            Aq = self.bspline_basis(t_n, knots, self.degree, eps=self.eps)
            z_out32 = Aq @ P32
        return z_out32.to(out_dtype)

    def spline_fit_and_eval(self, z_in: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
        P32, knots, t0, t1 = self.spline_fit(z_in, t_in)
        return self.spline_eval(P32, knots, t_out, t0, t1, out_dtype=z_in.dtype)

    # -------------------------
    # Forward
    # -------------------------

    def forward(
        self,
        in_frames: torch.Tensor,       # (B, T_in, C, L)
        in_timestamps: torch.Tensor,   # (B, T_in)
        out_timestamps: torch.Tensor,  # (B, T_out)
    ):
        """
        Returns:
          recon_out: (B, T_out, C, L)
          z_reg:     scalar (latent-to-spline L2)
        """
        _, _, _, L = in_frames.shape

        # Autoencode -> latent per step
        z_in = self.encode(in_frames)

        # Spline fit
        P32, knots, t0, t1 = self.spline_fit(z_in, in_timestamps)

        # Eval & decode
        z_out = self.spline_eval(P32, knots, out_timestamps, t0, t1, out_dtype=z_in.dtype)
        recon_out = self.decode(z_out, L=L)

        # Regularize: match spline at observed timestamps
        z_in_spline = self.spline_eval(P32, knots, in_timestamps, t0, t1, out_dtype=z_in.dtype)
        z_reg = (z_in - z_in_spline).pow(2).sum(dim=-1).mean()

        return recon_out, z_reg


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent = 128
    B, T, C, L = 32, 64, 3, 274

    x = torch.randn(B, T, C, L, device=device)
    timestamps = torch.linspace(0, 1, steps=T, device=device).unsqueeze(0).repeat(B, 1)

    model = SplineAutoEncoder1D(latent=latent, in_dim=C).to(device)

    # Profile memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=acts,
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        recon, z_reg = model(x, timestamps, timestamps)

    sort_key = f"self_{device}_memory_usage"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=8))

    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

    print(
        "Total trainable parameters:",
        round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2),
        "M",
    )
    print("recon:", tuple(recon.shape), "z_reg:", z_reg.item())
