import torch
from torch import nn
from SpatialTemporalPerceiver import sinusoidal_embedding_1d


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
