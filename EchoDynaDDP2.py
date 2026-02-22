# torchrun --nproc_per_node=4 EchoDynaDDP.py
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.SplineAutoEncoder import SplineAutoEncoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from datahandling.augmentations.get_augmentations import get_pretrain_augmentations
from datahandling.collate import AE_collate


# -------------------- DDP SETUP --------------------
def ddp_setup():
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not is_distributed:
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, rank, world_size, device


is_distributed, rank, world_size, device = ddp_setup()
is_main = (rank == 0)

# Make output_dir identical across ranks
date = None
timestamp = None
if is_main:
    date = datetime.now().strftime("%Y_%m_%d")
    timestamp = datetime.now().strftime("%H_%M")
if is_distributed:
    obj = [(date, timestamp)]
    dist.broadcast_object_list(obj, src=0)
    date, timestamp = obj[0]

output_dir = f"results/{date}/{timestamp}_SAE"
if is_main:
    os.makedirs(output_dir, exist_ok=True)

config = json.load(open("config/SAE2.json", "r"))

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

# Wrap in DDP
if is_distributed:
    model = DDP(model, device_ids=[device.index], output_device=device.index)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["training"]["epochs"]
)

# Load only train dataset (val removed)
train_ds, _, _ = load_echonet_dynamic_datasets(get_mask=False)

# Divide global batch size by number of GPUs
global_bs = int(config["training"]["batch_size"])
if is_distributed:
    if global_bs % world_size != 0:
        raise ValueError(
            f'config["training"]["batch_size"]={global_bs} must be divisible by WORLD_SIZE={world_size}'
        )
    per_gpu_bs = global_bs // world_size
else:
    per_gpu_bs = global_bs

# Distributed sampler (train only)
train_sampler = (
    DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    if is_distributed
    else None
)

train_dl = DataLoader(
    train_ds,
    batch_size=per_gpu_bs,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    drop_last=True,
    num_workers=20,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=lambda x: AE_collate(
        x,
        max_frames=config["training"]["max_frames"],
        augmentations=get_pretrain_augmentations()
        if config["training"]["augmentations"]
        else None,
    ),
)

epochs = int(config["training"]["epochs"])
autocast = bool(config["training"].get("autocast", True))
grad_clip = float(config["training"].get("grad_clip_max_norm", 1.0))
save_every = bool(config["training"].get("save_every_epoch", True))
use_aug = bool(config["training"].get("use_augmented_input", True))

criterion = nn.MSELoss()

train_losses: list[float] = []
z_regs: list[float] = []

if is_main:
    trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"SplineAutoEncoder: {trainable_m:.2f}M trainable parameters")


def _unwrap_for_saving(m: nn.Module) -> nn.Module:
    # Handles DDP + compile
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, "_orig_mod"):  # torch.compile
        m = m._orig_mod
    return m


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_unwrap_for_saving(model).state_dict(), path)


def save_loss_plot(
    train_losses: list[float],
    train_z_regs: list[float],
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x_tr = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x_tr, train_losses, label="Train Recon", color="blue", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_yscale("log")

    ax2 = ax.twinx()
    ax2.plot(x_tr, train_z_regs, label="Train Z Reg", color="orange", linestyle="-")
    ax2.set_ylabel("Z Regularization")

    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ddp_sum_(vals: list[float]) -> list[float]:
    if not is_distributed:
        return vals
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


for epoch in range(epochs):
    if is_distributed:
        train_sampler.set_epoch(epoch)

    # -------------------- TRAIN --------------------
    model.train()
    running = 0.0
    z_reg_running = 0.0
    seen = 0.0

    iterable = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}") if is_main else train_dl
    for batch in iterable:
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

        bs = float(A_frames.size(0))
        running += float(recon_loss.item()) * bs
        z_reg_running += float(z_reg.item()) * bs
        seen += bs

        if is_main:
            iterable.set_postfix(
                {"Recon": float(recon_loss.item()), "z_reg": float(z_reg.item()), "gnorm": float(gnorm)}
            )

    # Reduce epoch stats across GPUs
    running, z_reg_running, seen = ddp_sum_([running, z_reg_running, seen])

    epoch_loss = running / max(1.0, seen)
    epoch_z_reg = z_reg_running / max(1.0, seen)
    train_losses.append(epoch_loss)
    z_regs.append(epoch_z_reg)
    scheduler.step()

    if is_main:
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- train loss: {epoch_loss:.6f} - train z_reg: {epoch_z_reg:.6f}"
        )

        if save_every:
            save_loss_plot(
                train_losses, z_regs,
                os.path.join(output_dir, "losses.png"),
            )
            save_checkpoint(model, os.path.join(output_dir, "SAE.pth"))

# Save config & history json (rank 0 only)
if is_main:
    history = {
        "train_total": train_losses,
        "z_reg": z_regs,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Model and loss plot saved to {output_dir}")

if is_distributed:
    dist.destroy_process_group()