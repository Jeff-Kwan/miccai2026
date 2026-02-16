import torch
from torch import nn
import torch.nn.functional as F
from torch import autocast


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GroupNorm(4, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convs(x)

class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.activated = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.GELU())
        self.residual = nn.Sequential(
            nn.AvgPool2d(3, 2, 1),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        self.norm = nn.GroupNorm(4, out_ch)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.activated(x) + self.residual(x))


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.activated = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2, 0),
            nn.GELU())
        self.residual = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.norm = nn.GroupNorm(4, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.activated(x) + self.residual(x))


class SimpleConvEncoder(nn.Module):
    """
    Input:  x of shape [B, T, in_dim, H, W]
    Output: z of shape [B, T, latent]  (one vector per frame)
    """

    def __init__(self, latent: int, in_dim: int = 3):
        super().__init__()
        base = latent // 2

        # Channel schedule: inverse of decoder.
        channels = [base // 32, base // 16, base // 8, base // 4, base // 2, base, base]

        self.in_conv = nn.Conv2d(in_dim, channels[0], 3, 1, 1)
        
        # Build stages with loop: 2 ResBlocks + downsample each
        self.stages = nn.ModuleList()
        for i in range(1, len(channels)):
            stage = nn.Sequential(
                Downsample(channels[i-1], channels[i]),
                ResBlock(channels[i]),
                ResBlock(channels[i]))
            self.stages.append(stage)

        self.to_latent = nn.Conv2d(channels[-1], latent, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_mlp = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, latent*4),
            nn.GELU(),
            nn.Linear(latent*4, latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        h = self.in_conv(x)
        for stage in self.stages:
            h = stage(h)

        h = self.to_latent(h)            # (B*T, latent, hH, hW)
        h = self.pool(h)                 # (B*T, latent, 1, 1)
        z = h.reshape(B, T, -1)             # (B, T, latent)
        z = self.latent_mlp(z)
        return z


class SimpleConvDecoder(nn.Module):
    def __init__(self, latent: int, out_dim: int = 3):
        super().__init__()
        base = latent // 2

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, base, 2, 2, 0),     # 1 -> 2
            Upsample(base,      base // 2),                # 2 -> 4
            Upsample(base // 2, base // 4),                # 4 -> 8
            Upsample(base // 4, base // 8),                # 8 -> 16
            Upsample(base // 8, base // 16),               # 16 -> 32
            Upsample(base // 16, base // 32),              # 32 -> 64
            nn.Conv2d(base // 32, out_dim, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # z: (B, T, latent)
        B, T, latent = z.shape
        z = z.reshape(B * T, latent, 1, 1)                  # (B*T, latent, 1, 1)
        x = self.decoder(z)                              # (B*T, out_dim, 64, 64) nominal
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.reshape(B, T, -1, H, W)                    # (B, T, out_dim, H, W)




class SplineAutoEncoder(nn.Module):
    """
    Autoencoder + cubic B-spline smoothing in latent space.

    Spline fit uses a *second-difference* penalty on control points:
        argmin_P ||A P - Z||^2 + lam ||D P||^2
    where D is the (n_ctrl-2) x n_ctrl second-difference matrix.
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
        self.encoder = SimpleConvEncoder(latent=latent, in_dim=in_dim)
        self.decoder = SimpleConvDecoder(latent=latent, out_dim=out_dim)

        self.degree = degree
        self.n_ctrl = n_ctrl
        self.lam = lam
        self.eps = eps

    # -------------------------
    # Encode / decode
    # -------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, H, W) -> (B, T, latent)
        return self.encoder(x)

    def decode(self, z: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # (B, T, latent) -> (B, T, C, H, W)
        return self.decoder(z, H=H, W=W)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        z = self.encode(x)
        recon = self.decode(z, H=H, W=W)
        return recon, z

    # -------------------------
    # B-spline utilities
    # -------------------------

    @staticmethod
    def make_clamped_uniform_knots(t: torch.Tensor, degree: int, n_ctrl: int) -> torch.Tensor:
        """
        Clamped uniform knot vector per batch.
        t: (B, T)
        returns: (B, n_ctrl + degree + 1)
        """
        B = t.shape[0]
        t0 = t.min(dim=1).values
        t1 = t.max(dim=1).values

        n_knots = n_ctrl + degree + 1
        n_int = n_knots - 2 * (degree + 1)  # number of *interior* knots

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

        # Use a fp32-safe tolerance (ignore caller eps for this equality)
        tol = 10 * torch.finfo(t.dtype).eps
        last_knot = knots[:, -1][:, None]                    # (B, 1)
        is_last = t >= (last_knot - tol)                     # (B, T) robust "at end"

        # If exactly at the end, assign to the last interval for p=0
        N = torch.where(is_last[:, :, None], torch.zeros_like(N), N)
        N[:, :, -1] = N[:, :, -1] + is_last.to(t.dtype)

        def safe_div(num, den):
            return torch.where(den.abs() > tol, num / den, torch.zeros_like(num))

        for p in range(1, degree + 1):
            denom1 = knots[:, None, p:K-1] - knots[:, None, :K-1-p]
            denom2 = knots[:, None, p+1:K] - knots[:, None, 1:K-p]

            w1 = safe_div(t_ - knots[:, None, :K-1-p], denom1)
            w2 = safe_div(knots[:, None, p+1:K] - t_, denom2)

            N = w1 * N[:, :, :K-1-p] + w2 * N[:, :, 1:K-p]

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
        DtD is (n_ctrl, n_ctrl), PSD.
        """
        if n_ctrl < 3:
            return torch.zeros((n_ctrl, n_ctrl), device=device, dtype=dtype)

        D = torch.zeros((n_ctrl - 2, n_ctrl), device=device, dtype=dtype)
        i = torch.arange(n_ctrl - 2, device=device)
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
        return D.transpose(0, 1) @ D  # (n_ctrl, n_ctrl)

    # -------------------------
    # Spline fit + eval
    # -------------------------

    def spline_fit_and_eval(
        self,
        z_in: torch.Tensor,      # (B, T_in, latent)
        t_in: torch.Tensor,      # (B, T_in)
        t_out: torch.Tensor,     # (B, T_out)
    ) -> torch.Tensor:
        """
        Fit spline to z_in over t_in and evaluate at t_out.
        Spline math always runs in fp32 regardless of autocast state.
        """
        orig_dtype = z_in.dtype
        device = z_in.device

        # Force fp32 math for numerical stability
        with autocast('cuda', enabled=False):
            z_in32 = z_in.float()
            t_in32 = t_in.float()
            t_out32 = t_out.float()

            # Normalize time to [0, 1] for better conditioning (knots are in this range)
            t0 = torch.cat([t_in32, t_out32], dim=1).min(dim=1, keepdim=True).values
            t1 = torch.cat([t_in32, t_out32], dim=1).max(dim=1, keepdim=True).values
            t_in32 = (t_in32 - t0) / (t1 - t0 + self.eps)
            t_out32 = (t_out32 - t0) / (t1 - t0 + self.eps)

            knots = self.make_clamped_uniform_knots(t_in32, self.degree, self.n_ctrl)

            A  = self.bspline_basis(t_in32,  knots, self.degree, eps=self.eps)
            Aq = self.bspline_basis(t_out32, knots, self.degree, eps=self.eps)

            At = A.transpose(1, 2)
            AtA = At @ A
            AtZ = At @ z_in32

            DtD = self.second_difference_gram(
                self.n_ctrl, device=device, dtype=torch.float32
            )

            lhs = AtA + self.lam * DtD[None, :, :]

            P = torch.linalg.solve(lhs, AtZ)

            z_out32 = Aq @ P

        # Return to original dtype for the rest of the network
        return z_out32.to(orig_dtype)

    def forward_spline(
        self,
        in_frames: torch.Tensor,       # (B, T_in, C, H, W)
        in_timestamps: torch.Tensor,   # (B, T_in)
        out_timestamps: torch.Tensor,  # (B, T_out)
    ):
        """
        Returns:
          recon_out: (B, T_out, C, H, W)
          z_in:      (B, T_in, latent)
          z_out:     (B, T_out, latent)
        """
        B, T_in, C, H, W = in_frames.shape

        z_in = self.encode(in_frames)
        z_out = self.spline_fit_and_eval(
            z_in=z_in,
            t_in=in_timestamps,
            t_out=out_timestamps)
        recon_out = self.decode(z_out, H=H, W=W)
        return recon_out, z_in, z_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent = 512
    B, T, C, H, W = 32, 64, 3, 112, 112

    x = torch.randn(B, T, C, H, W, device=device)
    model = SplineAutoEncoder(latent=latent, in_dim=C).to(device)

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
        recon, z = model(x)

    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))

    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

    print(
        "Total trainable parameters:",
        round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2),
        "M",
    )
    print("recon:", tuple(recon.shape), "z:", tuple(z.shape))