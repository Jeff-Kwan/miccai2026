import torch
from torch import nn
import torch.nn.functional as F
    
class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, x):
        return x + self.convs(x)

class ConvEncoder(nn.Module):
    def __init__(self, in_c=3, latent=512, layers=4, levels=6):
        super(ConvEncoder, self).__init__()
        init_c = latent // (2 ** levels)
        self.in_conv = nn.Conv2d(in_c, init_c, 3, 1, 1)

        level_blocks = {}
        for level in range(levels):
            level_blocks[f"{level}"] = nn.Sequential(
                *[ConvBlock(init_c * (2 ** level))
                    for _ in range(layers)],
                nn.GroupNorm(1, init_c * (2 ** level)),
                nn.Conv2d(init_c*(2**level), init_c*(2** (level + 1)),
                    2, 2, 0))
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
    def __init__(self, out_c=1, latent=512, layers=4, levels=6):
        super(ConvDecoder, self).__init__()
        init_c = latent // (2 ** levels)
        level_blocks = {}
        for level in reversed(range(levels)):
            level_blocks[f"{level}"] = nn.Sequential(
                nn.ConvTranspose2d(init_c * (2 ** (level + 1)),
                    init_c * (2 ** level), 2, 2, 0),
                nn.GroupNorm(1, init_c * (2 ** level)),
                *[ConvBlock(init_c * (2 ** level))
                    for _ in range(layers)])
        self.levels = levels
        self.level_blocks = nn.ModuleDict(level_blocks)
        self.merges = nn.ModuleDict({
            f"{level}": nn.Conv2d(init_c * (2 ** (level + 2)),
                init_c * (2 ** (level+1)), 1, 1, 0)
            for level in reversed(range(levels))
        })
        self.out_conv = nn.Conv2d(init_c, out_c, 3, 1, 1)

    def forward(self, x, skips):
        for level in reversed(range(self.levels)):
            x = self.merges[f"{level}"](torch.cat([x, skips[level]], dim=1))
            x = self.level_blocks[f"{level}"](x)
        x = self.out_conv(x)
        return x


class ConvSegNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, latent=512, enc_layers=6, dec_layers=2, levels=6):
        super(ConvSegNet, self).__init__()
        self.latent = latent
        self.encoder = ConvEncoder(in_c, latent, enc_layers, levels)
        self.bottleneck = nn.Sequential(
            *[ConvBlock(latent) for _ in range(enc_layers)])
        self.decoder = ConvDecoder(out_c, latent, dec_layers, levels)

    def forward(self, x):
        z, skips = self.encoder(x)
        z = self.bottleneck(z)
        x = self.decoder(z, skips)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvSegNet(in_c=3, out_c=1, latent=256, enc_layers=6, dec_layers=4, levels=4).to(device)
    x = torch.randn(1, 3, 112, 112, device=device)
    with torch.no_grad():
        y = model(x)

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


    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1048**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1048 / 1048, 'MB')
    print("I/O has elements: ", round(output.nelement() / 1e6, 2), 'M')

