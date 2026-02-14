import os
import json
from datetime import datetime
from math import ceil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.SplineAutoEncoder import SplineAutoEncoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from datahandling.augmentations.get_augmentations import get_pretrain_augmentations
from datahandling.collate import AE_collate


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_loss_plot(losses: list[float], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = range(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, losses, label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%Y_%m_%d")
timestamp = datetime.now().strftime("%H_%M")
output_dir = f"results/{date}/{timestamp}_SAE"
os.makedirs(output_dir, exist_ok=True)

config = json.load(open("config/SAE.json", "r"))

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision(config["training"].get("matmul_precision", "high"))

mcfg = config["model"]
model = SplineAutoEncoder(
    latent=mcfg["latent"],
    in_dim=mcfg.get("in_dim", 3),
    out_dim=mcfg.get("out_dim", None),
    n_ctrl=ceil(config["training"]["max_frames"]//3)+3,
    degree=3,
    lam=1e-3,
).to(device)

if config["training"].get("compile", False):
    model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["training"]["epochs"]
)


train_ds, val_ds, _ = load_echonet_dynamic_datasets(get_mask=False)
train_dl = DataLoader(
    train_ds,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=24,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda x: AE_collate(
        x,
        max_frames=config["training"]["max_frames"],
        augmentations=get_pretrain_augmentations(),
    ),
)

epochs = int(config["training"]["epochs"])
autocast = bool(config["training"].get("autocast", True))
grad_clip = float(config["training"].get("grad_clip_max_norm", 1.0))
save_every = bool(config["training"].get("save_every_epoch", True))
use_aug = bool(config["training"].get("use_augmented_input", True))

criterion = nn.MSELoss()
train_losses: list[float] = []

trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"SplineAutoEncoder: {trainable_m:.2f}M trainable parameters")

for epoch in range(epochs):
    model.train()
    running = 0.0
    seen = 0

    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        in_frames = batch["in_frames"].to(device, non_blocking=True)
        out_frames = batch["out_frames"].to(device, non_blocking=True)
        in_timestamps = batch["in_timestamps"].to(device, non_blocking=True)
        out_timestamps = batch["out_timestamps"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            recon, z_in, z_out = model.forward_spline(in_frames, in_timestamps, out_timestamps)
            loss = criterion(recon, out_frames)

        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += loss.item() * config["training"]["batch_size"]
        seen += config["training"]["batch_size"]
        pbar.set_postfix({"loss": float(loss.item()), "gnorm": float(gnorm)})

    epoch_loss = running / max(1, seen)
    train_losses.append(epoch_loss)
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")

    if save_every:
        save_loss_plot(train_losses, os.path.join(output_dir, "losses.png"))
        save_checkpoint(model, os.path.join(output_dir, "SAE.pth"))

history = {"train_total": train_losses}
with open(os.path.join(output_dir, "history.json"), "w") as f:
    json.dump(history, f, indent=2)