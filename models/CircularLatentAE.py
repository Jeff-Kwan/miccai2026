import torch
from torch import nn

class SpatioTemporalConvBlock(nn.Module):
    def __init__(self, channels):
        super(SpatioTemporalConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)))

    def forward(self, x):
        return x + self.convs(x)
    
class SpatioConvBlock(nn.Module):
    def __init__(self, channels):
        super(SpatioConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

    def forward(self, x):
        return x + self.convs(x)

class ConvEncoder(nn.Module):
    def __init__(self, in_c=3, latent=512, layers=4, levels=6):
        super(ConvEncoder, self).__init__()
        init_c = latent // (2 ** levels)
        self.in_conv = nn.Conv3d(in_c, init_c, (1, 3, 3), 1, (0, 1, 1))

        level_blocks = {}
        for level in range(levels):
            level_blocks[f"{level}"] = nn.Sequential(
                *[SpatioTemporalConvBlock(init_c * (2 ** level))
                    for _ in range(layers)],
                nn.BatchNorm3d(init_c * (2 ** level)),
                nn.Conv3d(init_c*(2**level), init_c*(2** (level + 1)),
                    (1, 2, 2), (1, 2, 2), 0))
        self.levels = levels
        self.level_blocks = nn.ModuleDict(level_blocks)

    def forward(self, x):
        x = self.in_conv(x)
        for level in range(self.levels):
            x = self.level_blocks[f"{level}"](x)
        x = torch.mean(x, dim=(-2, -1), keepdim=True)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, out_c=3, latent=512, layers=4, levels=6):
        super(ConvDecoder, self).__init__()
        init_c = latent // (2 ** levels)
        self.in_conv = nn.Sequential(
            nn.Conv3d(latent, latent*2, 1, 1, 0),
            nn.GELU(),
            nn.ConvTranspose3d(latent*2, latent, (1, 2, 2), (1, 2, 2), 0))
        level_blocks = {}
        for level in reversed(range(levels)):
            level_blocks[f"{level}"] = nn.Sequential(
                nn.ConvTranspose3d(init_c * (2 ** (level + 1)),
                    init_c * (2 ** level), (1, 2, 2), (1, 2, 2), 0),
                nn.BatchNorm3d(init_c * (2 ** level)),
                *[SpatioConvBlock(init_c * (2 ** level))
                    for _ in range(layers)])
        self.levels = levels
        self.level_blocks = nn.ModuleDict(level_blocks)
        self.out_conv = nn.Conv3d(init_c, out_c, (1, 3, 3), 1, (0, 1, 1))

    def forward(self, x):
        x = self.in_conv(x)
        for level in reversed(range(self.levels)):
            x = self.level_blocks[f"{level}"](x)
        x = self.out_conv(x)
        return x


class CircularLatentAE(nn.Module):
    def __init__(self, in_c=3, latent=512, enc_layers=6, dec_layers=2, levels=6):
        super(CircularLatentAE, self).__init__()
        self.latent = latent
        self.encoder = ConvEncoder(in_c, latent, enc_layers, levels)
        self.decoder = ConvDecoder(in_c, latent, dec_layers, levels)

        self.centroid_mlp = nn.Sequential(
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, latent))
        self.motion_mlp = nn.Sequential(
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, 3))
        self.learnable_basis = nn.Parameter(torch.randn(latent, 4))

    def forward(self, x):
        B, C, T, H, W = x.shape
        z = self.encoder(x) # [B, latent, T, 1, 1]
        z = z.view(B, self.latent, T).transpose(1, 2)  # [B, T, latent]

        # Static Anatomical Structure
        z_centroid = self.centroid_mlp(z).mean(dim=1, keepdim=True)  # [B, 1, latent]

        # Circular phase motion latent
        motion = self.motion_mlp(z) # [B, T, 3]
        phase = motion[:, :, :1]
        deformation = motion[:, :, 1:]
        self.deformationL2 = deformation.pow(2).mean()
        modulation = torch.cat([torch.sin(phase), torch.cos(phase), deformation], dim=-1)  # [B, T, 2]
        
        Q, R = torch.linalg.qr(self.learnable_basis + 1e-8)
        delta_z = modulation @ Q.transpose(0, 1)  # [B, T, latent]

        z_hat = (z_centroid + delta_z).transpose(0, 1).reshape(B, self.latent, T, 1, 1)
        x_rec = self.decoder(z_hat)
        return x_rec


if __name__ == "__main__":
    model = CircularLatentAE(in_c=3, latent=512, enc_layers=4, dec_layers=2, levels=6)
    x = torch.randn(2, 3, 10, 128, 128)  # [B, C, T, H, W]
    with torch.no_grad():
        x_rec = model(x)
    print("Input shape:", x.shape)
    print("Reconstructed shape:", x_rec.shape)
    print("Number of parameters:", round(sum(p.numel() for p in model.parameters())/1e6, 2), "M")
    print(f"Deformation L2:", model.deformationL2.item())