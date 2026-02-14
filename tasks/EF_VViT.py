import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from models.VideoViT import VideoViTEncoder, VideoViTCfg
from models.Downstream import EF_Probe
from datahandling.collate import EF_collate
from datahandling.augmentations.get_augmentations import get_pretrain_augmentations
import matplotlib.pyplot as plt 
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_13/16_24_VMAE"
output_dir = os.path.join(load_dir, "EF_estimation")
os.makedirs(output_dir, exist_ok=True)

config = json.load(open("config/VMAE.json", "r"))
max_frames = config["training"]["max_frames"]
epochs = 100
batch_size = 32
lr = 2e-4
weight_decay = 1e-3
dropout = 0.1
torch_compile = True
torch.set_float32_matmul_precision('high')

enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
pretrained_dict = torch.load(os.path.join(load_dir, "VMAE.pth"), map_location=device)
enc.load_state_dict({k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if k.startswith("encoder.")})
probe = EF_Probe(encoder=enc, dropout=dropout).to(device)
print(f"Initialized EF Probe with {sum(p.numel() for p in probe.parameters() if p.requires_grad)/1e3:.2f}K trainable parameters.")
if torch_compile:
    probe = torch.compile(probe)
autocast = config["training"]["autocast"]

enc = list(probe.encoder.parameters())
optimizer = torch.optim.AdamW([
    {"params": enc, "lr": lr/10, "weight_decay": weight_decay/10},
    {"params": [p for p in probe.parameters() if id(p) not in {id(p) for p in enc}],
     "lr": lr, "weight_decay": weight_decay}])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
mse = nn.MSELoss()
l1 = nn.L1Loss()
smoothl1 = nn.SmoothL1Loss(beta=0.1)

# Dataset
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=False)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      collate_fn=lambda x: EF_collate(x, max_frames=max_frames, augmentations=get_pretrain_augmentations(), generator=None),
                      num_workers=24, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                    collate_fn=lambda x: EF_collate(x, max_frames=max_frames, augmentations=None, generator=None))
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                    collate_fn=lambda x: EF_collate(x, max_frames=max_frames, augmentations=None, generator=None))

for epoch in range(epochs):
    probe.train(); probe.encoder.eval()  # freeze encoder
    train_loss = 0.0
    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in p_bar:
        videos, ef, timestamps = batch["video"], batch["EF"], batch["timestamps"]
        videos, ef, timestamps = videos.to(device), ef.to(device), timestamps.to(device)
        optimizer.zero_grad()
        pred_ef = probe(videos, timestamps, autocast=autocast)
        loss = smoothl1(pred_ef, ef)
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
            pred_ef = probe(videos, timestamps, autocast=autocast)
            ef = ef * 100.0  # Denormalize EF to original scale
            pred_ef = pred_ef * 100.0  # Denormalize EF to original scale
            loss = l1(pred_ef, ef)
            rmse = mse(pred_ef, ef)
            val_loss += loss.item() * videos.size(0)
            val_rmse += rmse.item() * videos.size(0)
    val_loss /= len(val_dl.dataset)
    val_rmse /= len(val_dl.dataset)
    val_rmse = val_rmse ** 0.5

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val MAE Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

    if torch_compile:
        torch.save(probe._orig_mod.state_dict(), os.path.join(output_dir, "EF_Probe.pth"))
    else:
        torch.save(probe.state_dict(), os.path.join(output_dir, "EF_Probe.pth"))

# Test
probe.eval()
test_loss = 0.0; test_rmse = 0.0
with torch.no_grad():
    for batch in tqdm(test_dl, desc="Testing"):
        videos, ef, timestamps = batch["video"], batch["EF"], batch["timestamps"]
        videos, ef, timestamps = videos.to(device), ef.to(device), timestamps.to(device)
        pred_ef = probe(videos, timestamps, autocast=autocast)
        ef = ef * 100.0  # Denormalize EF to original scale
        pred_ef = pred_ef * 100.0  # Denormalize EF to original scale
        loss = l1(pred_ef, ef)
        rmse = mse(pred_ef, ef)
        test_loss += loss.item() * videos.size(0)
        test_rmse += rmse.item() * videos.size(0)
    test_loss /= len(test_dl.dataset)
    test_rmse /= len(test_dl.dataset)
    test_rmse = test_rmse ** 0.5

print(f"Test MAE Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}")


# Save txt report
with open(os.path.join(output_dir, "EF_estimation.txt"), "w") as f:
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Weight Decay: {weight_decay}\n\n")

    f.write(f"Final Val MAE Loss: {val_loss:.4f}\n")
    f.write(f"Final Val RMSE: {val_rmse:.4f}\n\n")

    f.write(f"Final Test MAE Loss: {test_loss:.4f}\n")
    f.write(f"Final Test RMSE: {test_rmse:.4f}\n")