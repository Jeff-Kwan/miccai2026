import torch
from torch import nn
from torch.nn import functional as F

class EF_Probe(nn.Module):
    def __init__(self, encoder, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        latent_dim = encoder.cfg.dim
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.01)
        self.qnorm = nn.LayerNorm(latent_dim)
        self.attnpool = nn.MultiheadAttention(latent_dim, num_heads=6, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*2, 1))
        self.fc[-1].bias.data.fill_(0.556)

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, video, timestamp, autocast):
        # with torch.no_grad():
        with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
            B, T, C, H, W = video.shape
            N = (H // self.encoder.cfg.patch) * (W // self.encoder.cfg.patch)
            keep_idx = torch.arange(N, device=video.device)[None, None, :].expand(B, T, N)
            cls, _, _ = self.encoder(video, keep_idx=keep_idx, timestamps=timestamp)

        gcls, _ = self.attnpool(self.qnorm(self.query).repeat(B, 1, 1), cls, cls)
        # Prediction mlp head
        pred = self.fc(gcls).squeeze(-1).squeeze(-1)  # [B]
        return pred


class LV_Segmentation(nn.Module):
    def __init__(self, encoder, out_c=1, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        latent_dim = encoder.cfg.dim
        patch = encoder.cfg.patch
        self.out_c = out_c
        self.fc = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*2, patch**2 * out_c))

        for param in self.encoder.parameters():
            param.requires_grad = False
        # self.conv_out = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Conv2d(latent_dim, latent_dim//4, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Conv2d(latent_dim//4, latent_dim//16, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(latent_dim//16, out_c, 2, 2, 0))

    def forward(self, video, timestamp, autocast):
        # with torch.no_grad():
        with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
            B, T, C, H, W = video.shape
            N = (H // self.encoder.cfg.patch) * (W // self.encoder.cfg.patch)
            keep_idx = torch.arange(N, device=video.device)[None, None, :].expand(B, T, N)
            cls, frames, _ = self.encoder(video, keep_idx=keep_idx, timestamps=timestamp)
            
        B, T, N, D = frames.shape
        pred = self.fc(frames)

        # Unpatchify and reassemble
        h, w = H // self.encoder.cfg.patch, W // self.encoder.cfg.patch
        pred = pred.view(B, T, h, w, self.encoder.cfg.patch, self.encoder.cfg.patch, self.out_c)
        pred = pred.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, T, self.out_c, H, W)
        # patches = patches.permute(0, 1, 3, 2).reshape(B*T, D, h, w)
        # pred = self.conv_out(patches).reshape(B, T, self.out_c, H, W)
        return pred