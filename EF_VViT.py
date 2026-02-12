import torch 
from torch import dtype, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datahandling.PreTrainEchoDynaDataset2 import load_echodyna_downstream_datasets
from models.VideoViT2 import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg
from models.ViTMAEMotion2 import VideoMotionMAE, SimpleConvDecoder
from datahandling.collate import EF_collate
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_11/17_24_VMAE"

max_frames = 64
epochs = 50
batch_size = 16
lr = 3e-4
weight_decay = 1e-3
autocast = True
config = json.load(open("config/VMAE.json", "r"))
enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
dec = VideoViTDecoder(enc_dim=config["encoder"]["dim"], patch=config["encoder"]["patch"], 
                      in_chans=config["encoder"]["in_chans"], cfg=VideoViTDecCfg(**config["decoder"]))
frame_dec = SimpleConvDecoder(latent=config["encoder"]["dim"], out_dim=config["encoder"]["in_chans"], base=config["decoder"]["dec_dim"])
mae = VideoMotionMAE(enc, dec, frame_dec, motion_dim=2, norm_pix_loss=False, mask_ratio=0.75)
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

    def forward(self, video, timestamp):
        with torch.no_grad():
            B, T, C, H, W = video.shape
            N = (H // self.encoder.cfg.patch) * (W // self.encoder.cfg.patch)
            keep_idx = torch.arange(N, device=device)[None, None, :].expand(B, T, N)
            gcls, frames, hw = self.encoder(video, keep_idx=keep_idx, timestamps=timestamp)
            fcls = frames[:, :, 0, :]
            
        # Attention Selection
        features = self.attn_pool(gcls.unsqueeze(1), fcls, fcls)[0].squeeze(1)  # [B, D]
        pred = self.fc(features)
        return pred.squeeze(-1)

probe = EDESMLPProbe(latent_dim=384, mae=mae).to(device)
optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
mse = nn.MSELoss()
l1 = nn.L1Loss()

# Dataset
train_ds, val_ds, test_ds = load_echodyna_downstream_datasets(allow_missing_masks=False)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      collate_fn=lambda x: EF_collate(x, max_frames=max_frames, augmentations=None, generator=None),
                      num_workers=60, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True,
                    collate_fn=lambda x: EF_collate(x, max_frames=max_frames, augmentations=None, generator=None))
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

for epoch in range(epochs):
    probe.train(); probe.encoder.eval()  # freeze encoder
    train_loss = 0.0
    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in p_bar:
        videos, ef, timestamps = batch["video"], batch["EF"], batch["timestamps"]
        videos, ef, timestamps = videos.to(device), ef.to(device), timestamps.to(device)
        optimizer.zero_grad()
        with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
            pred_ef = probe(videos, timestamps)
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
        for batch in val_dl:
            videos, ef, timestamps = batch["video"], batch["EF"], batch["timestamps"]
            videos, ef, timestamps = videos.to(device), ef.to(device), timestamps.to(device)
            with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                pred_ef = probe(videos, timestamps)
                ef = ef * 100.0  # Denormalize EF to original scale
                pred_ef = pred_ef * 100.0  # Denormalize EF to original scale
                loss = l1(pred_ef, ef)
            rmse = torch.sqrt(mse(pred_ef, ef))
            val_loss += loss.item() * videos.size(0)
            val_rmse += rmse.item() * videos.size(0)
    val_loss /= len(val_dl.dataset)
    val_rmse /= len(val_dl.dataset)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val MAE Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

# Test on full videos
test_loss = 0.0; test_rmse = 0.0
with torch.no_grad():
    for batch in tqdm(test_dl, desc="Testing"):
        videos, ef, timestamps = batch["video"], batch["metadata"]["EF"], batch["timestamps"]
        videos, ef, timestamps = videos.to(device), ef.to(device), timestamps.to(device)
        pred_ef = probe(videos, timestamps)
        ef = ef * 100.0  # Denormalize EF to original scale
        pred_ef = pred_ef * 100.0  # Denormalize EF to original scale
        loss = l1(pred_ef, ef)
        rmse = torch.sqrt(mse(pred_ef, ef))
        test_loss += loss.item() * videos.size(0)
        test_rmse += rmse.item() * videos.size(0)
    test_loss /= len(test_dl.dataset)
    test_rmse /= len(test_dl.dataset)

print(f"Test MAE Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}")

# Save txt report
with open(os.path.join(load_dir, "EF_estimation.txt"), "w") as f:
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Weight Decay: {weight_decay}\n\n")

    f.write(f"Final Val MAE Loss: {val_loss:.4f}\n")
    f.write(f"Final Val RMSE: {val_rmse:.4f}\n\n")

    f.write(f"Final Test MAE Loss: {test_loss:.4f}\n")
    f.write(f"Final Test RMSE: {test_rmse:.4f}\n")