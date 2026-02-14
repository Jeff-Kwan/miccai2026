import torch
from torch import nn
from torch.nn import functional as F

class EF_Probe(nn.Module):
    def __init__(self, encoder, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.query = nn.Parameter(torch.randn(1, 1, encoder.cfg.dim) * 0.01)
        self.qnorm = nn.LayerNorm(encoder.cfg.dim)
        self.attn_pool = nn.MultiheadAttention(embed_dim=encoder.cfg.dim, num_heads=6, batch_first=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(encoder.cfg.dim),
            nn.Dropout(dropout),
            nn.Linear(encoder.cfg.dim, 1))
        self.fc[-1].bias.data.fill_(0.556)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, video, timestamp, autocast):
        with torch.no_grad():
            with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                B, T, C, H, W = video.shape
                N = (H // self.encoder.cfg.patch) * (W // self.encoder.cfg.patch)
                keep_idx = torch.arange(N, device=video.device)[None, None, :].expand(B, T, N)
                gcls, frames, _ = self.encoder(video, keep_idx=keep_idx, timestamps=timestamp)
                tokens = torch.cat([gcls.unsqueeze(1), frames[:, :, 0, :]], dim=1)
            
        # Attention Selection
        features = self.attn_pool(self.qnorm(self.query).repeat(B, 1, 1)*(32/T)**0.5, tokens, tokens, need_weights=False)[0]  # [B, 1, D]
        pred = self.fc(features).squeeze(1)
        return pred.squeeze(-1)
