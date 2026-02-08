import torch
from torch import nn
import torch.nn.functional as F

class CausalConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(CausalConv, self).__init__()
        self.conv = nn.Conv3d(in_c, out_c, (2, 3, 3), 1, (0, 1, 1))

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 1, 0))  # pad only the past in time dimension
        return self.conv(x)

class SpatioTemporalConvBlock(nn.Module):
    def __init__(self, channels):
        super(SpatioTemporalConvBlock, self).__init__()
        self.convs = nn.Sequential(
            CausalConv(channels, channels),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            CausalConv(channels, channels))

    def forward(self, x):
        return x + self.convs(x)
    
class SpatioConvBlock(nn.Module):
    def __init__(self, channels):
        super(SpatioConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

    def forward(self, x):
        return x + self.convs(x)

class ConvEncoder(nn.Module):
    def __init__(self, in_c=3, latent=512, layers=4, levels=6):
        super(ConvEncoder, self).__init__()
        init_c = latent // (2 ** levels)
        self.levels = levels
        self.in_conv = nn.Conv3d(in_c, init_c, (1, 3, 3), (1, 2, 2), (0, 1, 1))

        level_blocks = {}; downs = {}
        for level in range(levels):
            level_blocks[f"{level}"] = nn.Sequential(
                *[SpatioTemporalConvBlock(init_c * (2 ** level))
                    for _ in range(layers)],
                nn.GroupNorm(1, init_c * (2 ** level)))
            downs[f"{level}"] = nn.Conv3d(init_c*(2**level), init_c*(2** (level + 1)),
                    (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.level_blocks = nn.ModuleDict(level_blocks)
        self.downs = nn.ModuleDict(downs)

        self.bottleneck = nn.Sequential(
            *[SpatioTemporalConvBlock(latent) for _ in range(layers)] +\
              [nn.GroupNorm(1, latent)])

    def forward(self, x):
        x = self.in_conv(x)
        skips = []
        for level in range(self.levels):
            x = self.level_blocks[f"{level}"](x)
            skips.append(x)
            x = self.downs[f"{level}"](x)
        x = self.bottleneck(x)
        return x, skips

class ConvDecoder(nn.Module):
    def __init__(self, out_c=3, latent=512, layers=4, levels=6, skips=False):
        super(ConvDecoder, self).__init__()
        self.levels = levels
        init_c = latent // (2 ** levels)

        level_blocks = {}; ups = {}; merges = {}
        for level in reversed(range(levels)):
            ups[f"{level}"] = nn.ConvTranspose3d(init_c * (2 ** (level + 1)),
                    init_c * (2 ** level), (1, 2, 2), (1, 2, 2), 0)
            
            level_blocks[f"{level}"] = nn.Sequential(
                nn.GroupNorm(1, init_c * (2 ** level)),
                *[SpatioConvBlock(init_c * (2 ** level))
                    for _ in range(layers)])
            
            if skips:
                merges[f"{level}"] = nn.Conv3d(init_c * (2 ** (level+1)), 
                                            init_c * (2 ** level), 1, 1, 0)
        self.ups = nn.ModuleDict(ups)
        self.level_blocks = nn.ModuleDict(level_blocks)
        if skips:
            self.merges = nn.ModuleDict(merges)
        self.out_conv = nn.ConvTranspose3d(init_c, out_c, (1, 2, 2), (1, 2, 2), 0)

        

    def forward(self, x, skips=None):
        for level in reversed(range(self.levels)):
            x = self.ups[f"{level}"](x)
            if skips is not None:
                x = torch.cat([x, skips.pop()], dim=1)
                x = self.merges[f"{level}"](x)
            x = self.level_blocks[f"{level}"](x)
        x = self.out_conv(x)
        return x


class MotionLatentAE(nn.Module):
    def __init__(self, in_c=3, out_c=3, latent=512, enc_layers=6, dec_layers=2, levels=6, skips=False):
        super(MotionLatentAE, self).__init__()
        self.latent = latent
        self.levels = levels
        self.skips = skips
        self.encoder = ConvEncoder(in_c, latent, enc_layers, levels)
        self.decoder = ConvDecoder(out_c, latent, dec_layers, levels, skips=skips)

        self.down = nn.Conv3d(latent, latent*2, (1, 2, 2), (1, 2, 2), 0)
        self.up = nn.ConvTranspose3d(latent*2, latent, (1, 2, 2), (1, 2, 2), 0)

    def svdvals_fp32(self, A):
        return torch.linalg.svdvals(A.float()).to(A.dtype)

    @torch.no_grad()
    def batch_effective_rank(self, s, eps: float = 1e-12):
        e = s.square()
        p = e / (e.sum(dim=-1, keepdim=True) + eps)
        p_safe = p.clamp_min(eps)
        H = -(p * p_safe.log()).sum(dim=-1)
        return H.exp()


    def spectral_entropy_penalty(self, s: torch.Tensor, eps: float = 1e-12):
        w = s.square()
        p = w / (w.sum(dim=-1, keepdim=True) + eps)
        p_safe = p.clamp_min(eps)
        H = -(p * p_safe.log()).sum(dim=-1)
        return H.mean()


    def schatten_p_mean_power(self, s: torch.Tensor, p: float = 1.0, eps: float = 1e-12):
        if p <= 0:
            return (s.clamp_min(eps).pow(p).mean(dim=-1)).mean()
        else:
            return (s.pow(p).mean(dim=-1)).mean()

        
    def forward(self, x):
        z, skips = self.encoder(x)

        z = self.down(z)
        v = z - z.mean(dim=2, keepdim=True)
        self.v = v.squeeze()    # [B, latent, T]
        s = self.svdvals_fp32(self.v)
        self.effective_rank = self.batch_effective_rank(s).mean()
        if self.training:
            self.latent_reg = self.spectral_entropy_penalty(s) +\
                                self.schatten_p_mean_power(s, p=1.0)
        z = self.up(z)

        if self.skips:
            return self.decoder(z, skips)
        else:
            return self.decoder(z, skips=None)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLatentAE(in_c=3, out_c=3, latent=256, enc_layers=4, 
                           dec_layers=2, levels=5, skips=False)
    model = model.to(device)
    x = torch.randn(32, 3, 64, 128, 128, device=device)  # [B, C, T, H, W]

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

    assert output.shape == x.shape, f"Output shape {output.shape} does not match input shape {x.shape}"
    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1024 / 1024, 'MB')
    print("I/O has elements: ", round(output.nelement() / 1e6, 2), 'M')