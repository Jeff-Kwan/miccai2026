import torch
from torch import nn
from .SpatialTemporalPerceiver import SpatialTemporalLatentPerceiver, sinusoidal_embedding_1d
    
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
    def __init__(self, in_c=3, init_c=8, latent=128, layers=4, levels=3):
        super(ConvEncoder, self).__init__()
        self.in_conv = nn.Conv3d(in_c, init_c, 1, 1, 0)

        level_blocks = {}
        for level in range(levels):
            level_blocks[f"{level}"] = nn.Sequential(
                *[SpatialConvBlock(init_c * (2 ** level))
                    for _ in range(layers)],
                nn.GroupNorm(4, init_c * (2 ** level)),
                nn.Conv3d(init_c*(2**level), init_c*(2** (level + 1)) if level<levels-1 else latent,
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
    def __init__(self, out_c=3, init_c=8, latent=128, layers=4, levels=3, skips=False):
        super(ConvDecoder, self).__init__()
        self.skips = skips
        level_blocks = {}
        for level in reversed(range(levels)):
            level_blocks[f"{level}"] = nn.Sequential(
                nn.ConvTranspose3d(init_c * (2 ** (level + 1)) if level<levels-1 else latent,
                    init_c * (2 ** level), (1, 2, 2), (1, 2, 2), 0),
                nn.GroupNorm(4, init_c * (2 ** level)),
                *[SpatialConvBlock(init_c * (2 ** level))
                    for _ in range(layers)])
        self.levels = levels
        self.level_blocks = nn.ModuleDict(level_blocks)
        self.out_conv = nn.Conv3d(init_c, out_c, 1, 1, 0)

        if self.skips:
            self.merges = nn.ModuleDict({
                f"{level}": nn.Conv3d(init_c * (2 ** (level+2)) if level < levels -1 else latent*2,
                                      init_c * (2 ** (level+1)) if level < levels -1 else latent, 1, 1, 0)
                for level in range(levels)})

    def forward(self, x, skips=None):
        for level in reversed(range(self.levels)):
            if self.skips:
                x = torch.cat([x, skips[level]], dim=1)
                x = self.merges[f"{level}"](x)
            x = self.level_blocks[f"{level}"](x)
        x = self.out_conv(x)
        return x

class INRDecoder(nn.Module):
    def __init__(self, in_c=128, hidden_c=128, out_c=3, layers=4):
        super(INRDecoder, self).__init__()
        self.pos_dim = in_c
        block = [nn.Conv3d(in_c*2, hidden_c, kernel_size=1, bias=False)]
        block.append(nn.GELU())
        for _ in range(layers - 2):
            block.append(nn.Conv3d(hidden_c, hidden_c, 1, 1, 0))
            block.append(nn.GELU())
        block.append(nn.Conv3d(hidden_c, out_c, 1, 1, 0))
        self.network = nn.Sequential(*block)

    def build_frame_pos(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        # 2D spatial positional embedding (x, y) only
        Ch = self.pos_dim // 2
        Cw = self.pos_dim - Ch
        h = sinusoidal_embedding_1d(H, Ch, device)
        w = sinusoidal_embedding_1d(W, Cw, device)
        h = h[:, None, :].expand(H, W, Ch)
        w = w[None, :, :].expand(H, W, Cw)
        pos = torch.cat([h, w], dim=-1)          # [H,W,P]
        pos = pos.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1,P,1,H,W]
        return pos

    def forward(self, x):
        B, C, T, H, W = x.shape
        pos = self.build_frame_pos(H, W, x.device)
        x = torch.cat([x.expand(-1, -1, -1, H, W), pos.expand(B, -1, T, -1, -1)], dim=1)
        x = self.network(x)
        return x

class MotionLatentPerceiver(nn.Module):
    def __init__(self, in_c=3, out_c=3, init_c=8, latent=512, enc_layers=6, t_layers=8, t_heads=4, t_latents=2, 
                 dec_layers=2, levels=6, motion_dim=2, masking_ratio=0.75, skips=False):
        super(MotionLatentPerceiver, self).__init__()
        assert t_latents >= 2, "Need at least 2 latents: one for centroid, one for motion"
        assert motion_dim >= 2, "Motion dimension should be at least 2D"
        self.latent = latent
        self.skips = skips
        self.encoder = ConvEncoder(in_c, init_c, latent, enc_layers, levels)
        self.perceiver = SpatialTemporalLatentPerceiver(
                dim=latent,
                depth=t_layers,
                num_heads=t_heads,
                num_latents=t_latents,
                mlp_ratio=4.0,
                keep_ratio=1.0 - masking_ratio,
                do_masking=not skips)
        self.INR_decoder = INRDecoder(in_c=latent, hidden_c=latent, out_c=latent, layers=4)
        self.conv_decoder = ConvDecoder(out_c, init_c, latent, dec_layers, levels, skips)
        self.centroid_mlp = nn.Sequential(
            nn.Linear(latent, latent*2),
            nn.GELU(),
            nn.Linear(latent*2, latent))
        self.motion_mlp = nn.Sequential(
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, motion_dim, bias=False))
        self.motion_basis = nn.Parameter(torch.randn(latent, motion_dim) / latent**0.5)

    def set_masking(self, do_masking: bool):
        self.perceiver.do_masking = do_masking

    def forward(self, x):
        B, C, T, H, W = x.shape

        # Convolutional embedding
        x, skips = self.encoder(x) # [B, latent, T, H', W']
        _, _, _, h, w = x.shape

        # Spatial-Temporal Latent Perceiver
        z = self.perceiver(x)   # [B, T, N-latents, latent]]

        # First vector is centroid
        z_centroid = self.centroid_mlp(z[:, :, 0, :].mean(dim=[1], keepdim=True))

        # Second vector is motion
        z_motion = self.motion_mlp(z[:, :, 1, :])
        # First 2 dimensions are on unit circle
        xy = z_motion[:, :, :2]
        xy_norm = xy / (xy.norm(dim=-1, keepdim=True) + 1e-8)
        z_motion = torch.cat([xy_norm, z_motion[:, :, 2:]], dim=-1)
        
        # self.z_motion = z_motion    # Visualization
        Q, R = torch.linalg.qr(self.motion_basis + 1e-8, mode='reduced')
        delta_z = z_motion @ Q.T
        
        z = (z_centroid + delta_z).transpose(1, 2).unsqueeze(-1).unsqueeze(-1).contiguous()
        z = z.expand(-1, -1, -1, h, w)  # [B, latent, T, H', W']
        z_centroid = z_centroid.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).contiguous()
        z_centroid = z_centroid.expand(-1, -1, -1, h, w)

        x_rec = self.conv_decoder(self.INR_decoder(z), skips if self.skips else None)
        x_centroid = self.conv_decoder(self.INR_decoder(z_centroid), skips=None).expand(-1, -1, T, -1, -1) if not self.skips else None
        return x_rec, x_centroid


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLatentPerceiver(in_c=3, init_c=4, out_c=3, latent=256, 
                           enc_layers=2, t_layers=12, t_heads=4, t_latents=2,
                            dec_layers=2, levels=4, skips=True)
    model = model.to(device)
    x = torch.randn(8, 3, 64, 128, 128, device=device)  # [B, C, T, H, W]

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
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1024 / 1024, 'MB')
    print("I/O has elements: ", round(output.nelement() / 1e6, 2), 'M')