import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datahandling.EchoDynaDataset import load_echonet_dynamic_datasets
from models.ConvSegNet import ConvSegNet
import os
import random
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from time import time


# --------------------
# Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%Y_%m_%d")
timestamp = datetime.now().strftime("%H_%M")
output_dir = f"results/{date}/{timestamp}_ConvSegNet"
os.makedirs(output_dir, exist_ok=True)

comments = [
    "Simple ConvSegNet segmentation training",
    "BCEWithLogitsLoss, AdamW, CosineAnnealingLR",
]

# Training Parameters
epochs = 500
batch_size = 32
learning_rate = 2e-4
weight_decay = 1e-3

train_params = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
}

augmentations = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomApply([
        v2.RandomAffine(
            degrees=30,
            translate=(0.1, 0.1),
            scale=(0.9, 1.2),
            interpolation=InterpolationMode.BILINEAR,
        ),
    ], p=0.5),
    v2.RandomApply([
        v2.GaussianBlur(kernel_size=7, sigma=(0.25, 1.5))
    ], p=0.4),
    v2.RandomApply([
        v2.GaussianNoise(0, 0.01)
    ], p=0.4),
    v2.RandomApply([
        v2.ColorJitter(brightness=0.2, contrast=0.2)
    ], p=0.4)
])


# --------------------
# Metrics
# --------------------
def _safe_div_tensor(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return num / (den + eps)


@torch.inference_mode()
def calculate_metrics_from_logits(logits: torch.Tensor, masks: torch.Tensor):
    preds = (logits >= 0).to(torch.int64)
    targets = (masks >= 0.5).to(torch.int64)

    # (N, H*W)
    p = preds.flatten(1)
    t = targets.flatten(1)

    tp = ((p == 1) & (t == 1)).sum(dim=1).to(torch.float32)
    tn = ((p == 0) & (t == 0)).sum(dim=1).to(torch.float32)
    fp = ((p == 1) & (t == 0)).sum(dim=1).to(torch.float32)
    fn = ((p == 0) & (t == 1)).sum(dim=1).to(torch.float32)

    precision = _safe_div_tensor(tp, tp + fp)
    recall    = _safe_div_tensor(tp, tp + fn)
    accuracy  = _safe_div_tensor(tp + tn, tp + tn + fp + fn)
    jaccard   = _safe_div_tensor(tp, tp + fp + fn)
    dice      = _safe_div_tensor(2 * tp, 2 * tp + fp + fn)

    return {
        "jaccard":   float(jaccard.mean().item()),
        "precision": float(precision.mean().item()),
        "recall":    float(recall.mean().item()),
        "accuracy":  float(accuracy.mean().item()),
        "dice":      float(dice.mean().item()),
    }


# --------------------
# Plotting / Examples
# --------------------
def save_examples_echo_dyna(model, val_ds, out_dir, results=5):
    """EchoDynaDataset format example saver (similar to your old trainer)."""
    results = min(results, len(val_ds))
    random_indices = random.sample(range(len(val_ds)), results)

    fig, axes = plt.subplots(results, 3, figsize=(18, 6 * results))
    if results == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    for i, idx in enumerate(random_indices):
        item = val_ds[idx]
        frames = item["tracing"]["frames"]
        masks = item["tracing"]["masks"]

        frame = frames[0]
        mask = masks[0]  # (1,H,W) presumably

        image = item["video"][:, frame, :, :].unsqueeze(0).to(device)  # (1,3,H,W)
        with torch.inference_mode():
            output = model(image)  # logits

        pred = torch.sigmoid(output).detach().squeeze().cpu().numpy()   # (H,W)
        true_mask = mask.detach().cpu().squeeze().numpy()               # (H,W)

        img_np = image.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
        mn, mx = img_np.min(), img_np.max()
        if mn >= -1.1 and mx <= 1.1:
            img_disp = ((img_np + 1) * 127.5).astype(np.uint8)
        elif mx <= 1.01:
            img_disp = (img_np * 255).astype(np.uint8)
        else:
            img_disp = np.clip(img_np, 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(img_disp)

        predicted_int = (pred * 255).astype(np.uint8)
        predicted_pil = Image.fromarray(predicted_int)

        h, w = pred.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 1] = (((pred > 0.5) & (true_mask == 1)) * 255).astype(np.uint8)  # TP -> Green
        overlay[:, :, 0] = (((pred > 0.5) & (true_mask == 0)) * 255).astype(np.uint8)  # FP -> Red
        overlay[:, :, 0] += (((pred <= 0.5) & (true_mask == 1)) * 255).astype(np.uint8)  # FN -> Amber
        overlay[:, :, 1] += (((pred <= 0.5) & (true_mask == 1)) * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay)

        axes[i, 0].imshow(image_pil)
        axes[i, 0].set_title("Original Image", fontsize=18)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(predicted_pil, cmap="gray")
        axes[i, 1].set_title("Predicted Probabilities", fontsize=18)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay_pil)
        axes[i, 2].set_title("Overlay (G:T, Y:FN, R:FP)", fontsize=18)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "examples.png"), bbox_inches="tight")
    plt.close(fig)


def plot_losses_and_dice(train_losses, val_losses, dice_list, out_dir):
    epochs_axis = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.plot(epochs_axis, train_losses, label="Train Loss")
    ax1.plot(epochs_axis, val_losses, label="Val Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Dice")
    ax2.plot(epochs_axis, dice_list, label="Dice")
    ax2.tick_params(axis="y")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("Losses and Dice Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "losses.png"))
    plt.close(fig)


def write_results_txt(out_dir, model, model_size, train_params, comments, best_results, start_time, test_results=None):
    elapsed_time = time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(f"Model size: {model_size/1e6:.3f} M\n")
        f.write(f"Training time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n\n")
        for c in comments:
            f.write(f"{c}\n")

        if hasattr(model, "model_params"):
            f.write(f"\nModel params: {json.dumps(model.model_params, indent=4)}\n")
        else:
            f.write("\nModel params: (model has no attribute 'model_params')\n")

        f.write(f"\nTrain params: {json.dumps(train_params, indent=4)}\n")
        f.write(f"\nBest validation results: {json.dumps(best_results, indent=4)}\n")
        if test_results is not None:
            f.write("\n~~~~~~ Test Results ~~~~~~\n")
            f.write(f"\nTest results: {json.dumps(test_results, indent=4)}\n")


# --------------------
# Test
# --------------------
def test_model(model, test_dl):
    model.eval()

    metric_sums = {"jaccard": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "dice": 0.0}
    num_images = 0

    with torch.inference_mode():
        for batch in tqdm(test_dl, desc="Testing"):
            imgs = batch["imgs"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)

            logits = model(imgs)
            batch_metrics = calculate_metrics_from_logits(logits, masks)

            bs = imgs.size(0)
            num_images += bs
            for k in metric_sums.keys():
                metric_sums[k] += batch_metrics[k] * bs  # weight by number of images

    return {k: (metric_sums[k] / max(1, num_images)) for k in metric_sums.keys()}


# --------------------
# Dataset + Dataloaders
# --------------------
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(
    "data/echodyna/FileList.csv",
    "data/echodyna/Videos",
    "data/echodyna/VolumeTracings.csv",
    load_video=True
)

def collate_fn(batch, apply_augs):
    imgs = []
    masks = []
    for item in batch:
        frames = item["tracing"]["frames"]
        targets = item["tracing"]["masks"]
        for frame, target in zip(frames, targets):
            imgs.append(item["video"][:, frame, :, :])
            masks.append(target)
    imgs = torch.stack(imgs)
    masks = torch.stack(masks).float()
    masks = (masks > 0.5).float()
    if apply_augs:
        imgs, masks = augmentations(imgs, masks)
    return {"imgs": imgs, "masks": masks}

def train_collate_fn(batch):
    return collate_fn(batch, apply_augs=True)

def val_collate_fn(batch):
    return collate_fn(batch, apply_augs=False)

train_dl = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn,
    num_workers=16, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_collate_fn,
    num_workers=16)
test_dl = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=val_collate_fn,
    num_workers=16)


# --------------------
# Model + Optim
# --------------------
model = ConvSegNet(in_c=3, out_c=1, latent=256, enc_layers=6, dec_layers=4, levels=4).to(device)
model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Initialized ConvSegNet with {model_size/1e6:.2f}M trainable parameters.")

focal_loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        B = logits.size(0)
        probs = torch.sigmoid(logits)
        probs = probs.view(B, -1)
        targets = targets.view(B, -1)
        intersection = (probs * targets).sum(dim=-1)
        total = probs.sum(dim=-1) + targets.sum(dim=-1)
        dice_loss = (1 - (2. * intersection + self.smooth) / (total + self.smooth)).mean()
        return dice_loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(f"logits shape {logits.shape} and target shape {target.shape} must match.")
        target = target.float()

        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")

        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")

dice_loss = DiceLoss()
focal_loss = FocalLoss()


# --------------------
# Training
# --------------------
train_losses = []
val_losses = []
val_metrics = {
    "jaccard": [],
    "precision": [],
    "recall": [],
    "accuracy": [],
    "dice": [],
}

best_results = {
    "epoch (count from 0)": None,
    "train_losses": None,
    "val_losses": None,
    "val_metrics": None,
}
best_dice = -1.0

start_time = time()

for epoch in range(epochs):
    # ---- Train ----
    model.train()
    train_loss_sum = 0.0
    num_train_samples = 0

    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in p_bar:
        imgs = batch["imgs"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)

        focal = focal_loss(logits, masks)
        dloss = dice_loss(logits, masks)
        loss = focal + dloss

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = imgs.size(0)
        train_loss_sum += loss.item() * bs
        num_train_samples += bs

        p_bar.set_postfix({"Focal Loss": focal.item(), "Dice Loss": dloss.item(), "Grad Norm": float(norm)})

    train_loss = train_loss_sum / max(1, num_train_samples)
    train_losses.append(train_loss)

    # ---- Validate (loss + metrics) ----
    model.eval()
    val_loss_sum = 0.0
    num_val_samples = 0

    # FIXED: accumulate per-image metrics weighted by number of images
    metric_sums = {"jaccard": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "dice": 0.0}
    num_val_images = 0

    with torch.inference_mode():
        p_bar = tqdm(val_dl, desc=f"Validation {epoch+1}/{epochs}")
        for batch in p_bar:
            imgs = batch["imgs"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)

            logits = model(imgs)
            focal = focal_loss(logits, masks)
            dloss = dice_loss(logits, masks)
            loss = focal + dloss

            bs = imgs.size(0)
            val_loss_sum += loss.item() * bs
            num_val_samples += bs

            batch_metrics = calculate_metrics_from_logits(logits, masks)
            num_val_images += bs
            for k in metric_sums.keys():
                metric_sums[k] += batch_metrics[k] * bs

            p_bar.set_postfix({"Focal Loss": focal.item(), "Dice Loss": dloss.item()})

    val_loss = val_loss_sum / max(1, num_val_samples)
    val_losses.append(val_loss)

    # FIXED: dataset-level mean = (sum over images) / (num images)
    precision = metric_sums["precision"] / max(1, num_val_images)
    recall    = metric_sums["recall"]    / max(1, num_val_images)
    accuracy  = metric_sums["accuracy"]  / max(1, num_val_images)
    jaccard   = metric_sums["jaccard"]   / max(1, num_val_images)
    dice      = metric_sums["dice"]      / max(1, num_val_images)

    val_metrics["precision"].append(precision)
    val_metrics["recall"].append(recall)
    val_metrics["accuracy"].append(accuracy)
    val_metrics["jaccard"].append(jaccard)
    val_metrics["dice"].append(dice)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Dice: {dice:.5f}")

    # ---- Save model (always) ----
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

    # ---- Save best model ----
    if dice >= best_dice:
        best_dice = dice
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        best_results = {
            "epoch (count from 0)": epoch,
            "train_losses": train_loss,
            "val_losses": val_loss,
            "val_metrics": {
                "jaccard": jaccard,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "dice": dice,
            },
        }

    # ---- Save metrics.json ----
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics": val_metrics,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # ---- Save plots + qualitative examples ----
    save_examples_echo_dyna(model, val_ds, output_dir, results=5)
    plot_losses_and_dice(train_losses, val_losses, val_metrics["dice"], output_dir)

    # ---- Save results.txt ----
    write_results_txt(
        output_dir,
        model=model,
        model_size=model_size,
        train_params=train_params,
        comments=comments,
        best_results=best_results,
        start_time=start_time)

# --------------------
# Final Test + Write Results
# --------------------
test_metrics = test_model(model, test_dl)
write_results_txt(
    output_dir,
    model=model,
    model_size=model_size,
    train_params=train_params,
    comments=comments,
    best_results=best_results,
    start_time=start_time,
    test_results=test_metrics)