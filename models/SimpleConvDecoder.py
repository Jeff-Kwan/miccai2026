import torch
from torch import nn
import torch.nn.functional as F

class SimpleConvDecoder(nn.Module):
    """
    Input:  z of shape [B, T, latent]  (CLS tokens per frame)
    Output: x of shape [B, T, out_dim, H, W]
    """
    def __init__(self, latent: int, out_dim: int = 3, base: int = 256):
        super().__init__()

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.GELU(),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, base, 2, 2, 0),     # 1 -> 2
            up_block(base,      base // 2),                # 2 -> 4
            up_block(base // 2, base // 4),                # 4 -> 8
            up_block(base // 4, base // 8),                # 8 -> 16
            up_block(base // 8, base // 16),               # 16 -> 32
            up_block(base // 16, base // 32),              # 32 -> 64
            nn.Conv2d(base // 32, out_dim, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, H, W) -> torch.Tensor:
        # z: (B, T, latent)
        B, T, latent = z.shape
        z = z.view(B * T, latent, 1, 1)                  # (B*T, latent, 1, 1)
        x = self.decoder(z)                              # (B*T, out_dim, 64, 64) nominal
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.view(B, T, -1, H, W)                    # (B, T, out_dim, H, W)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent = 512
    B, T, C, H, W = 32, 64, 3, 112, 112
    x = torch.randn(B, T, latent, device=device)
    model = SimpleConvDecoder(latent=latent, out_dim=C, base=latent//2).to(device)

    # Profile memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None,
        ],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        out = model(x, H=H, W=W)

    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    print(
        "Total trainable parameters:",
        round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2),
        "M",
    )
