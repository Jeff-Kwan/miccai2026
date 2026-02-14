import torch
from torch import nn
from torch.nn import functional as F

class EF_Probe(nn.Module):
    def __init__(self, encoder, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        latent_dim = encoder.cfg.dim
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
            gcls, frames, _ = self.encoder(video, keep_idx=keep_idx, timestamps=timestamp)
            
        # Attention Selection
        pred = self.fc(gcls).squeeze(-1)  # [B,T]
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

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, video, timestamp, autocast):
        # with torch.no_grad():
        with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
            B, T, C, H, W = video.shape
            N = (H // self.encoder.cfg.patch) * (W // self.encoder.cfg.patch)
            keep_idx = torch.arange(N, device=video.device)[None, None, :].expand(B, T, N)
            gcls, frames, _ = self.encoder(video, keep_idx=keep_idx, timestamps=timestamp)
            patches = frames[:, :, 1:, :]  # [B, T, N, D]
            
        B, T, N, D = patches.shape
        pred = self.fc(patches)

        # Unpatchify and reassemble
        h, w = H // self.encoder.cfg.patch, W // self.encoder.cfg.patch
        pred = pred.view(B, T, h, w, self.encoder.cfg.patch, self.encoder.cfg.patch, self.out_c)
        pred = pred.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, T, self.out_c, H, W)
        return pred