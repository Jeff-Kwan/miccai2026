import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datahandling.PreTrainEchoDynaDataset import load_echodyna_downstream_datasets
from models.VideoViT import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg
from models.ViTMAEMotion import VideoMotionMAE, SimpleConvDecoder
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_10/15_52_VMAE"
output_dir = os.path.join(load_dir, "reconstructions")
os.makedirs(output_dir, exist_ok=True)

max_frames = 32
epochs = 30
batch_size = 8
enc = VideoViTEncoder(VideoViTCfg(dim=384, depth=8, heads=6, patch=8))
dec = VideoViTDecoder(enc_dim=384, patch=8, in_chans=3, cfg=VideoViTDecCfg(dec_dim=256, dec_depth=2, dec_heads=8))
frame_dec = SimpleConvDecoder(latent=384, out_dim=3, base=256)
mae = VideoMotionMAE(enc, dec, frame_dec, motion_dim=2, norm_pix_loss=True, mask_ratio=0.75)
mae.load_state_dict(torch.load(os.path.join(load_dir, "VMAE.pth"), map_location=device))
mae = mae.to(device)
mae.eval()

for param in mae.parameters():
    param.requires_grad = False

class EDESMLPProbe(nn.Module):
    def __init__(self, latent_dim, mae):
        super().__init__()
        self.encoder = mae.encoder
        self.attn_pool = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=6, batch_first=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1))
        self.fc[1].bias.data.fill_(0.556)

    def forward(self, video):
        with torch.no_grad():
            gcls, frames, hw = self.encoder(video)
            fcls = frames[:, :, 0, :]
            
        # Attention Selection
        features = self.attn_pool(gcls.unsqueeze(1), fcls, fcls)[0].squeeze(1)  # [B, D]
        pred = self.fc(features)
        return pred.squeeze(-1)

probe = EDESMLPProbe(latent_dim=384, mae=mae).to(device)
optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
mse = nn.MSELoss()
l1 = nn.L1Loss()

# Dataset
def collate_fn(batch):
    # Random sample to the minimum number of frames in the batch, with max cap
    min_frames = min(min(item['video'].shape[1] for item in batch), max_frames)
    for item in batch:
        T = item['video'].shape[1]
        if T > min_frames:
            max_start = T - min_frames
            start = torch.randint(0, max_start + 1, (1,)).item()
            item['video'] = item['video'][:, start:start + min_frames, :, :]
        else:
            item['video'] = item['video'][:, :min_frames, :, :]
    videos = torch.stack([item['video'] for item in batch])  # [B, C, T, H, W]
    videos = videos.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]

    # collect EF
    ef = [sample["metadata"]["EF"] for sample in batch]
    ef = torch.tensor(ef, dtype=torch.float32)
    ef = ef / 100.0  # Normalize EF to [0, 1]
    return videos, ef

train_ds, val_ds, test_ds = load_echodyna_downstream_datasets(allow_missing_masks=False)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=60, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True,
                    collate_fn=collate_fn)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True,
                     collate_fn=collate_fn)

for epoch in range(epochs):
    probe.train(); probe.encoder.eval()  # freeze encoder
    train_loss = 0.0
    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for videos, ef in p_bar:
        videos, ef = videos.to(device), ef.to(device)
        optimizer.zero_grad()
        pred_ef = probe(videos)
        loss = l1(pred_ef, ef)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * videos.size(0)
        p_bar.set_postfix({"Loss": loss.item(), "GradNorm": norm.item()})

    scheduler.step()
    train_loss /= len(train_dl.dataset)

    probe.eval()
    val_loss = 0.0; val_rmse = 0.0
    with torch.no_grad():
        for videos, ef in val_dl:
            videos, ef = videos.to(device), ef.to(device)
            pred_ef = probe(videos)
            ef = ef * 100.0  # Denormalize EF to original scale
            pred_ef = pred_ef * 100.0  # Denormalize EF to original scale
            loss = l1(pred_ef, ef)
            rmse = torch.sqrt(mse(pred_ef, ef))
            val_loss += loss.item() * videos.size(0)
            val_rmse += rmse.item() * videos.size(0)
    val_loss /= len(val_dl.dataset)
    val_rmse /= len(val_dl.dataset)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val MAE Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

test_loss = 0.0; test_rmse = 0.0
with torch.no_grad():
    for videos, ef in test_dl:
        videos, ef = videos.to(device), ef.to(device)
        pred_ef = probe(videos)
        ef = ef * 100.0  # Denormalize EF to original scale
        pred_ef = pred_ef * 100.0  # Denormalize EF to original scale
        loss = l1(pred_ef, ef)
        rmse = torch.sqrt(mse(pred_ef, ef))
        test_loss += loss.item() * videos.size(0)
        test_rmse += rmse.item() * videos.size(0)
    test_loss /= len(test_dl.dataset)
    test_rmse /= len(test_dl.dataset)

print(f"Test MAE Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}")