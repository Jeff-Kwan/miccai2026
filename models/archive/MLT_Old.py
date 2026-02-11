import torch
from torch import nn
    
class SpatialConvBlock(nn.Module):
    def __init__(self, channels):
        super(SpatialConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

    def forward(self, x):
        return x + self.convs(x)

class ConvEncoder(nn.Module):
    def __init__(self, in_c=3, latent=128, layers=4, levels=3):
        super(ConvEncoder, self).__init__()
        init_c = latent // (2 ** levels)
        self.in_conv = nn.Conv3d(in_c, init_c, (1, 3, 3), 1, (0, 1, 1))

        level_blocks = {}
        for level in range(levels):
            level_blocks[f"{level}"] = nn.Sequential(
                *[SpatialConvBlock(init_c * (2 ** level))
                    for _ in range(layers)],
                nn.GroupNorm(1, init_c * (2 ** level)),
                nn.Conv3d(init_c*(2**level), init_c*(2** (level + 1)),
                    (1, 2, 2), (1, 2, 2), 0))
        self.bottleneck = nn.Sequential(*[SpatialConvBlock(latent) for _ in range(layers)])
        self.levels = levels
        self.level_blocks = nn.ModuleDict(level_blocks)

    def forward(self, x):
        x = self.in_conv(x)
        skips = []
        for level in range(self.levels):
            x = self.level_blocks[f"{level}"](x)
            skips.append(x)
        x = self.bottleneck(x)
        return x, skips

class ConvDecoder(nn.Module):
    def __init__(self, out_c=3, latent=128, layers=4, levels=3, skips=False):
        super(ConvDecoder, self).__init__()
        init_c = latent // (2 ** levels)
        self.skips = skips
        level_blocks = {}
        for level in reversed(range(levels)):
            level_blocks[f"{level}"] = nn.Sequential(
                nn.ConvTranspose3d(init_c * (2 ** (level + 1)),
                    init_c * (2 ** level), (1, 2, 2), (1, 2, 2), 0),
                nn.GroupNorm(1, init_c * (2 ** level)),
                *[SpatialConvBlock(init_c * (2 ** level))
                    for _ in range(layers)])
        self.levels = levels
        self.level_blocks = nn.ModuleDict(level_blocks)
        self.out_conv = nn.Conv3d(init_c, out_c, (1, 3, 3), 1, (0, 1, 1))

        if self.skips:
            self.merges = nn.ModuleDict({
                f"{level}": nn.Conv3d(init_c * (2 ** (level+2)),
                                      init_c * (2 ** (level+1)), 1, 1, 0)
                for level in range(levels)})

    def forward(self, x, skips=None):
        for level in reversed(range(self.levels)):
            if self.skips:
                x = torch.cat([x, skips[level]], dim=1)
                x = self.merges[f"{level}"](x)
            x = self.level_blocks[f"{level}"](x)
        x = self.out_conv(x)
        return x


class MotionLatentAE(nn.Module):
    def __init__(self, in_c=3, out_c=3, latent=512, enc_layers=6, dec_layers=2, levels=6, motion_dim=2, skips=False):
        super(MotionLatentAE, self).__init__()
        self.latent = latent
        self.skips = skips
        self.encoder = ConvEncoder(in_c, latent, enc_layers, levels)
        self.decoder = ConvDecoder(out_c, latent, dec_layers, levels, skips)

        self.centroid_mlp = nn.Sequential(
            nn.Conv3d(latent, latent*2, 1, 1, 0),
            nn.GELU(),
            nn.Conv3d(latent*2, latent, 1, 1, 0))
        self.motion_mlp = nn.Sequential(
            nn.Conv3d(latent, latent*2, 1, 1, 0),
            nn.GELU(),
            nn.Conv3d(latent*2, motion_dim, 1, 1, 0))
        self.motion_basis = nn.Parameter(torch.randn(latent, motion_dim))

    def forward(self, x):
        B, C, T, H, W = x.shape
        z, skips = self.encoder(x) # [B, latent, T, H', W']

        # Spatially-static structural component
        z_centroid = self.centroid_mlp(z).mean(dim=2, keepdim=True)  # [B, latent, 1, H', W']

        # Frame-wise motion component
        z_motion = self.motion_mlp(z).mean(dim=[3,4])  # [B, latent, T, 1, 1]
        self.z_motion = z_motion    # Visualization
        Q, R = torch.linalg.qr(self.motion_basis + 1e-8, mode='reduced')
        delta_z = (Q @ z_motion).unsqueeze(-1).unsqueeze(-1)  # [B, latent, T, 1, 1]

        z_hat = z_centroid + delta_z
        x_rec = self.decoder(z_hat, skips if self.skips else None)
        x_centroid = self.decoder(z_centroid, skips=None).expand(-1, -1, T, -1, -1) if not self.skips else None
        return x_rec, x_centroid


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLatentAE(in_c=3, latent=512, enc_layers=4, dec_layers=2, levels=6, skips=True)
    model = model.to(device)
    x = torch.randn(6, 3, 64, 128, 128, device=device)  # [B, C, T, H, W]

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
        output, _ = model(x)

    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1048**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1048 / 1048, 'MB')
    print("I/O has elements: ", round(output.nelement() / 1e6, 2), 'M')