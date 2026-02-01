import torch
from torch import nn
from .SpatialTemporalPerceiver import SpatialTemporalLatentPerceiver
    
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
    def __init__(self, in_c=3, init_c=8, layers=4, levels=3):
        super(ConvEncoder, self).__init__()
        self.in_conv = nn.Conv3d(in_c, init_c, 1, 1, 0)

        level_blocks = {}
        for level in range(levels):
            level_blocks[f"{level}"] = nn.Sequential(
                *[SpatialConvBlock(init_c * (2 ** level))
                    for _ in range(layers)],
                nn.GroupNorm(4, init_c * (2 ** level)),
                nn.Conv3d(init_c*(2**level), init_c*(2** (level + 1)),
                    (1, 2, 2), (1, 2, 2), 0))
        self.levels = levels
        self.level_blocks = nn.ModuleDict(level_blocks)

    def forward(self, x):
        x = self.in_conv(x)
        skips = []
        for level in range(self.levels):
            x = self.level_blocks[f"{level}"](x)
            skips.append(x)
        return x, skips

class ConvDecoder(nn.Module):
    def __init__(self, out_c=3, init_c=8, layers=4, levels=3, skips=False):
        super(ConvDecoder, self).__init__()
        self.skips = skips
        level_blocks = {}
        for level in reversed(range(levels)):
            level_blocks[f"{level}"] = nn.Sequential(
                nn.ConvTranspose3d(init_c * (2 ** (level + 1)),
                    init_c * (2 ** level), (1, 2, 2), (1, 2, 2), 0),
                nn.GroupNorm(4, init_c * (2 ** level)),
                *[SpatialConvBlock(init_c * (2 ** level))
                    for _ in range(layers)])
        self.levels = levels
        self.level_blocks = nn.ModuleDict(level_blocks)
        self.out_conv = nn.Conv3d(init_c, out_c, 1, 1, 0)

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

class MotionLatentPerceiver(nn.Module):
    def __init__(self, in_c=3, out_c=3, init_c=8, latent=512, enc_layers=6, t_layers=8, t_heads=4, t_latents=2, 
                 dec_layers=2, levels=6, motion_dim=2, masking_ratio=0.75, skips=False):
        super(MotionLatentPerceiver, self).__init__()
        assert t_latents >= 3, "Need at least 3 latents"
        assert motion_dim >= 2, "Motion dimension should be at least 2D"
        self.latent = latent
        self.skips = skips
        self.encoder = ConvEncoder(in_c, init_c, enc_layers, levels)
        self.perceiver = SpatialTemporalLatentPerceiver(
                dim=latent,
                frame_dim=init_c * (2 ** levels),
                depth=t_layers,
                num_heads=t_heads,
                num_latents=t_latents,
                mlp_ratio=4.0,
                keep_ratio=1.0 - masking_ratio,
                do_masking=not skips)
        self.conv_decoder = ConvDecoder(out_c, init_c, dec_layers, levels, skips)

        self.template_mlp = nn.Sequential(
            nn.Linear(latent, latent*2),
            nn.GELU(),
            nn.Linear(latent*2, latent))
        self.motion_mlp = nn.Sequential(
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, motion_dim, bias=False))
        self.upsampler = nn.Sequential(
            nn.ConvTranspose3d(latent, init_c * (2 ** (levels+2)), (1, 2, 2), (1, 2, 2), 0),
            nn.GELU(),
            nn.ConvTranspose3d(init_c * (2 ** (levels+2)), init_c * (2 ** (levels+1)), (1, 2, 2), (1, 2, 2), 0),
            nn.GELU(),
            nn.ConvTranspose3d(init_c * (2 ** (levels+1)), init_c * (2 ** levels), (1, 2, 2), (1, 2, 2), 0),
            nn.GroupNorm(1, init_c * (2 ** levels)))

    def set_masking(self, do_masking: bool):
        self.perceiver.do_masking = do_masking

    def forward(self, x):
        B = x.shape[0]
        # Convolutional embedding
        x_lat, skips = self.encoder(x) # [B, latent, T, H', W']

        # Spatial-Temporal Latent Perceiver
        z = self.perceiver(x_lat)   # [B, T, N-latents, latent]]

        # Motion components mix static templates
        z_motion = self.motion_mlp(z[:, :, 0, :]).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        # First 2 dimensions are on unit circle
        # xy = z_motion[:, :, :2]
        # xy_norm = xy / (xy.norm(dim=-1, keepdim=True) + 1e-8)
        # z_motion = torch.cat([xy_norm, z_motion[:, :, 2:]], dim=-1)

        # Time-Averaged templates x2
        z_c1 = self.template_mlp(z[:, :, 1, :].mean(dim=[1], keepdim=True)).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        z_c2 = self.template_mlp(z[:, :, 2, :].mean(dim=[1], keepdim=True)).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        x_c1 = self.upsampler(z_c1)
        x_c2 = self.upsampler(z_c2)
        x_hat = z_motion[:, 0:1, :, :, :] * x_c1 + z_motion[:, 1:2, :, :, :] * x_c2
        
        # self.z_motion = z_motion    # Visualization

        x_rec = self.conv_decoder(x_hat, skips if self.skips else None)
        return x_rec,

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLatentPerceiver(in_c=3, out_c=3, init_c=8, latent=256, 
                              enc_layers=2, t_layers=12, t_heads=4, t_latents=4, 
                            dec_layers=2, levels=4, 
                            motion_dim=2,   # 2 templates
                            masking_ratio=0.75, skips=True)
    model = model.to(device)
    x = torch.randn(16, 3, 64, 128, 128, device=device)  # [B, C, T, H, W]

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
        output = model(x)[0]

    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1024 / 1024, 'MB')
    print("I/O has elements: ", round(output.nelement() / 1e6, 2), 'M')