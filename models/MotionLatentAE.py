from collections.abc import Sequence
import torch
from torch import nn
import torch.nn.functional as F

class SpatioTemporalConvBlock(nn.Module):
    def __init__(self, channels):
        super(SpatioTemporalConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, 1, 1))

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
                *[SpatioConvBlock(init_c * (2 ** level))
                    for _ in range(layers)],
                nn.GroupNorm(1, init_c * (2 ** level)))
            downs[f"{level}"] = nn.Conv3d(init_c*(2**level), init_c*(2** (level + 1)),
                    (1, 2, 2), (1, 2, 2), 0)
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
        skips.append(x)
        return skips

class ConvDecoder(nn.Module):
    def __init__(self, out_c=3, latent=512, layers=4, levels=6):
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
            
            merges[f"{level}"] = nn.Conv3d(init_c * (2 ** (level+1)), 
                                           init_c * (2 ** level), 1, 1, 0)
        self.ups = nn.ModuleDict(ups)
        self.level_blocks = nn.ModuleDict(level_blocks)
        self.merges = nn.ModuleDict(merges)
        self.out_conv = nn.ConvTranspose3d(init_c, out_c, (1, 2, 2), (1, 2, 2), 0)

        

    def forward(self, x, skips=None):
        # x = skips.pop()
        for level in reversed(range(self.levels)):
            x = self.ups[f"{level}"](x)
            if skips is not None:
                x = torch.cat([x, skips.pop()], dim=1)
                x = self.merges[f"{level}"](x)
            x = self.level_blocks[f"{level}"](x)
        x = self.out_conv(x)
        return x


class MotionLatentAE(nn.Module):
    def __init__(self, in_c=3, out_c=3, latent=512, enc_layers=6, dec_layers=2, levels=6, motion_dim=2):
        super(MotionLatentAE, self).__init__()
        self.latent = latent
        self.levels = levels
        self.encoder = ConvEncoder(in_c, latent, enc_layers, levels)
        self.decoder = ConvDecoder(out_c, latent, dec_layers, levels)

        # self.motion_mlp = nn.Sequential(
        #     nn.LayerNorm(latent),
        #     nn.Linear(latent, latent*2),
        #     nn.GELU(),
        #     nn.Linear(latent*2, motion_dim, bias=False))
        # self.motion_basis = nn.Parameter(torch.randn(latent, motion_dim) * 0.01)
        
        # self.centroid_mlps = nn.Sequential(
        #         nn.Conv3d(latent, latent*4, 1, 1, 0),
        #         nn.GELU(),
        #         nn.Conv3d(latent*4, latent, 1, 1, 0))
        self.down = nn.Conv3d(latent, latent*2, (1, 2, 2), (1, 2, 2), 0)
        self.up = nn.ConvTranspose3d(latent*2, latent, (1, 2, 2), (1, 2, 2), 0)

    def stable_rank_penalty(self,
        X: torch.Tensor,
        matrix_dims: Sequence[int],
        eps: float = 1e-12,
        reduction: str = "mean",
    ):
        """
        Stable rank penalty over a batch of matrices.
        """

        row_dim, col_dim = matrix_dims
        ndim = X.ndim
        row_dim %= ndim
        col_dim %= ndim

        if row_dim == col_dim:
            raise ValueError("matrix dimensions must be different")

        # Move matrix dims to end
        batch_dims = [d for d in range(ndim) if d not in (row_dim, col_dim)]
        perm = batch_dims + [row_dim, col_dim]
        Y = X.permute(*perm).contiguous()

        batch_shape = Y.shape[:-2]
        m, n = Y.shape[-2:]
        Y = Y.reshape(-1, m, n)  # (B, m, n)

        # Frobenius norm squared
        fro2 = (Y**2).sum(dim=(-2, -1))  # (B,)

        # Largest singular value
        smax = torch.linalg.svdvals(Y)[..., 0]  # (B,)
        spec2 = smax**2 + eps

        stable_rank = fro2 / spec2  # (B,)

        stable_rank = stable_rank.reshape(batch_shape)

        if reduction == "mean":
            return stable_rank.mean()
        elif reduction == "sum":
            return stable_rank.sum()
        elif reduction == "none":
            return stable_rank
        else:
            raise ValueError("Invalid reduction")
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        skips = self.encoder(x)

        z = skips.pop()
        z = self.down(z)
        v = z - z.mean(dim=2, keepdim=True)
        v = v / v.norm(dim=(1, 2), keepdim=True)
        self.latent_reg = self.stable_rank_penalty(v, matrix_dims=(1, 2))
        z = self.up(z)

        # # Frame-wise motion component
        # z_motion = self.motion_mlp(skips[-1].mean(dim=[3,4]).transpose(1, 2))  # [B, T, latent]
        # self.z_motion = z_motion    # Visualization

        # # for l in range(self.levels + 1):
        # # Orthonormal motion basis
        # Q, _ = torch.linalg.qr(self.motion_basis + 1e-8, mode='reduced')
        # motion = (z_motion @ Q.T).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        # # Static structural centroid
        # centroid = self.centroid_mlps(skips[-1].mean(dim=2, keepdim=True))

        # # Final features
        # skips[-1] = centroid + motion

        # for l in range(self.levels):
        #     skips[self.levels-1-l] = skips[self.levels-1-l].detach()

        x_rec = self.decoder(z, skips=None)
        return x_rec


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLatentAE(in_c=3, out_c=3, latent=256, enc_layers=4, 
                           dec_layers=2, levels=5, motion_dim=2)
    model = model.to(device)
    x = torch.randn(16, 3, 128, 128, 128, device=device)  # [B, C, T, H, W]

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