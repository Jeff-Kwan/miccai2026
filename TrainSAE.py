import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.SplineAutoEncoder import SplineAutoEncoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from datahandling.augmentations.get_augmentations import get_pretrain_augmentations
from datahandling.collate import AE_collate


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

model = SplineAutoEncoder(
    latent=config["model"]["latent"],
    in_dim=config["model"].get("in_dim", 3),
    out_dim=config["model"].get("out_dim", None),
    n_ctrl_params=config["model"]["n_ctrl_params"],
    degree=config["model"]["degree"],
    lam=config["model"]["lam"],
).to(device)

if config["training"].get("compile", False):
    model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["training"]["epochs"])


train_ds, val_ds, _ = load_echonet_dynamic_datasets(get_mask=False)

train_dl = DataLoader(
    train_ds,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=24,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=lambda x: AE_collate(
        x,
        max_frames=config["training"]["max_frames"],
        augmentations=get_pretrain_augmentations() if config["training"]["augmentations"] else None,
    ),
)

val_dl = DataLoader(
    val_ds,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=lambda x: AE_collate(
        x,
        max_frames=config["training"]["max_frames"],
        augmentations=None)
)

epochs = int(config["training"]["epochs"])
autocast = bool(config["training"].get("autocast", True))
grad_clip = float(config["training"].get("grad_clip_max_norm", 1.0))
save_every = bool(config["training"].get("save_every_epoch", True))
use_aug = bool(config["training"].get("use_augmented_input", True))

criterion = nn.MSELoss()

# --- CHANGED: track train + val recon and z_reg separately ---
train_losses: list[float] = []
z_regs: list[float] = []
val_losses: list[float] = []
val_z_regs: list[float] = []

trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"SplineAutoEncoder: {trainable_m:.2f}M trainable parameters")


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if config["training"].get("compile", False):
        torch.save(model._orig_mod.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

# --- CHANGED: plot train solid, val dotted; recon blue, reg orange ---
def save_loss_plot(
    train_losses: list[float],
    train_z_regs: list[float],
    val_losses: list[float],
    val_z_regs: list[float],
    path: str
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x_tr = range(1, len(train_losses) + 1)
    x_va = range(1, len(val_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Recon (blue)
    ax.plot(x_tr, train_losses, label="Train Recon", color="blue", linestyle="-")
    if len(val_losses) > 0:
        ax.plot(x_va, val_losses, label="Val Recon", color="blue", linestyle=":")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_yscale("log")

    # Z reg (orange) on twin axis
    ax2 = ax.twinx()
    ax2.plot(x_tr, train_z_regs, label="Train Z Reg", color="orange", linestyle="-")
    if len(val_z_regs) > 0:
        ax2.plot(x_va, val_z_regs, label="Val Z Reg", color="orange", linestyle=":")
    ax2.set_ylabel("Z Regularization")

    # Combine legends from both axes
    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


for epoch in range(epochs):
    # -------------------- TRAIN --------------------
    model.train()
    running = 0.0
    z_reg_running = 0.0
    seen = 0

    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        A_frames = batch["in_frames"].to(device)  # [B, T, C, H, W]
        A_timestamps = batch["in_timestamps"].to(device)  # [B, T]
        B_frames = batch["out_frames"].to(device)  # [B, T, C, H, W]
        B_timestamps = batch["out_timestamps"].to(device)  # [B, T]
        _, _, _, H, W = A_frames.shape

        all_frames_in = torch.cat([A_frames, B_frames], dim=0)
        all_frames_out = torch.cat([B_frames, A_frames], dim=0)
        all_t_in = torch.cat([A_timestamps, B_timestamps], dim=0)
        all_t_out = torch.cat([B_timestamps, A_timestamps], dim=0)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            # A-B partition cross-consistency
            recon, z_in, z_spline = model(all_frames_in, all_t_in, all_t_out)
            z_inA, z_inB = z_in.chunk(2, dim=0)
            z_splineA, z_splineB = z_spline.chunk(2, dim=0)

            recon_loss = criterion(recon, all_frames_out)
            z_reg = (z_inA - z_splineB).pow(2).sum(dim=-1).mean() + (z_inB - z_splineA).pow(2).sum(dim=-1).mean()
            loss = recon_loss + config["training"]["reg"] * z_reg

        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += recon_loss.item() * config["training"]["batch_size"]
        z_reg_running += z_reg.item() * config["training"]["batch_size"]
        seen += config["training"]["batch_size"]
        pbar.set_postfix({"Recon": float(recon_loss.item()), "z_reg": float(z_reg.item()), "gnorm": float(gnorm)})

    epoch_loss = running / max(1, seen)
    epoch_z_reg = z_reg_running / max(1, seen)
    train_losses.append(epoch_loss)
    z_regs.append(epoch_z_reg)
    scheduler.step()

    # -------------------- VAL (NEW) --------------------
    model.eval()
    val_running = 0.0
    val_z_reg_running = 0.0
    val_seen = 0

    with torch.no_grad():
        vbar = tqdm(val_dl, desc=f"Val {epoch+1}/{epochs}", leave=False)
        for batch in vbar:
            A_frames = batch["in_frames"].to(device)  # [B, T, C, H, W]
            A_timestamps = batch["in_timestamps"].to(device)  # [B, T]
            B_frames = batch["out_frames"].to(device)  # [B, T, C, H, W]
            B_timestamps = batch["out_timestamps"].to(device)  # [B, T]
            _, _, _, H, W = A_frames.shape

            all_frames_in = torch.cat([A_frames, B_frames], dim=0)
            all_frames_out = torch.cat([B_frames, A_frames], dim=0)
            all_t_in = torch.cat([A_timestamps, B_timestamps], dim=0)
            all_t_out = torch.cat([B_timestamps, A_timestamps], dim=0)
            
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                # A-B partition cross-consistency
                recon, z_in, z_spline = model(all_frames_in, all_t_in, all_t_out)
                z_inA, z_inB = z_in.chunk(2, dim=0)
                z_splineA, z_splineB = z_spline.chunk(2, dim=0)

                recon_loss = criterion(recon, all_frames_out)
                z_reg = (z_inA - z_splineB).pow(2).sum(dim=-1).mean() + (z_inB - z_splineA).pow(2).sum(dim=-1).mean()
                loss = recon_loss + config["training"]["reg"] * z_reg

            val_running += recon_loss.item() * config["training"]["batch_size"]
            val_z_reg_running += z_reg.item() * config["training"]["batch_size"]
            val_seen += config["training"]["batch_size"]
            vbar.set_postfix({"Recon": float(recon_loss.item()), "z_reg": float(z_reg.item())})

    epoch_val_loss = val_running / max(1, val_seen)
    epoch_val_z_reg = val_z_reg_running / max(1, val_seen)
    val_losses.append(epoch_val_loss)
    val_z_regs.append(epoch_val_z_reg)

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"- train loss: {epoch_loss:.6f} - train z_reg: {epoch_z_reg:.6f} "
        f"- val loss: {epoch_val_loss:.6f} - val z_reg: {epoch_val_z_reg:.6f}"
    )

    if save_every:
        save_loss_plot(
            train_losses, z_regs,
            val_losses, val_z_regs,
            os.path.join(output_dir, "losses.png")
        )
        save_checkpoint(model, os.path.join(output_dir, "SAE.pth"))

# --- CHANGED: include val in history ---
history = {
    "train_total": train_losses,
    "z_reg": z_regs,
    "val_total": val_losses,
    "val_z_reg": val_z_regs,
}

# Save config & history json
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
with open(os.path.join(output_dir, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"Training complete. Model and loss plot saved to {output_dir}")
