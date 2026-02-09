import torch 
from datahandling.PreTrainEchoDynaDataset import load_echodyna_downstream_datasets
from models.MotionLatentAE3 import MotionLatentAE
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2, InterpolationMode
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import json
from time import time
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_09/01_42_MLAE"
output_dir = os.path.join(load_dir, "LVSeg")
os.makedirs(output_dir, exist_ok=True)

model = MotionLatentAE(in_c=3, out_c=1, latent=256, enc_layers=4, 
                           dec_layers=2, levels=5, skips=True)
pretrained = torch.load(os.path.join(load_dir, "MLAE.pth"), map_location=device)
model_dict = model.state_dict()
matched = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(matched)
model.load_state_dict(model_dict)
model = model.to(device)
model.encoder.requires_grad_(False)
model.down.requires_grad_(False)
# model.centroid_mlps.requires_grad_(False)
# model.motion_mlp.requires_grad_(False)
model.motion_basis.requires_grad_(False)

# Training Parameters
epochs = 100
batch_size = 32
learning_rate = 3e-4
weight_decay = 1e-2
frames = 32

train_params = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
}

comments = [
]


class RandomGamma(nn.Module):
    def __init__(self, gamma=(0.7, 1.5)):
        super().__init__()
        self.gamma = gamma

    def forward(self, x):
        # assume x in [-1,1]
        x = (x + 1) * 0.5           # -> [0,1]
        g = torch.empty(1, device=x.device).uniform_(*self.gamma)
        return (x.pow(g) * 2 - 1).clamp(-1.0, 1.0)

class ClipBrightnessContrast(nn.Module):
    def __init__(self, brightness=0.3, contrast=0.2):
        super().__init__()
        self.b = brightness
        self.c = contrast

    def forward(self, x):
        # x: [B, T, C, H, W]
        b = torch.empty(1, device=x.device).uniform_(-self.b, self.b)
        c = torch.empty(1, device=x.device).uniform_(1 - self.c, 1 + self.c)
        mean = x.mean(dim=(-2, -1), keepdim=True)  # per B,T,C over H,W
        return ((x - mean) * c + mean + b).clamp(-1.0, 1.0)

class SpeckleNoise(torch.nn.Module):
    def __init__(self, std=(0.02, 0.1)):
        super().__init__()
        self.std = std

    def forward(self, x):
        # assume x in [-1,1]
        x = (x + 1) * 0.5           # -> [0,1]
        std = torch.empty(1, device=x.device).uniform_(*self.std)
        noise = torch.randn_like(x) * std
        x = (x + x * noise).clamp(0.0, 1.0)
        return x * 2 - 1            # back to [-1,1]

augmentations = v2.Compose([
    v2.RandomApply([# Intensities
        v2.RandomChoice([
            v2.RandomChoice([# Intensity distribution
                ClipBrightnessContrast(brightness=0.3, contrast=0.2),
                RandomGamma(gamma=(0.7, 1.5))]),
            v2.RandomChoice([# Sharpness / Blur
                v2.RandomAdjustSharpness(sharpness_factor=0.5, p=1),
                v2.GaussianBlur(kernel_size=7, sigma=(0.25, 1.5))]),
            v2.RandomChoice([# Noise
                v2.GaussianNoise(0, 0.05),
                SpeckleNoise(std=(0.02, 0.1))])
        ])
    ], p=0.5),
    v2.RandomApply([# Masking
        v2.RandomErasing(p=1)
    ], p=0.3),
])

model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Initialized ConvSegNet with {model_size/1e6:.2f}M trainable parameters.")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
        frames = item["frame_indices"]
        masks = item["masks"]

        frame = frames[0]
        mask = masks[0]  # (1,H,W) presumably

        image = item["video"][:, frame, :, :].unsqueeze(0).unsqueeze(2).to(device)  # (1,3,1,H,W)
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
            idx = batch["keyframe_idx"]

            logits = model(imgs)
            logits = torch.stack([logits_i[:, idx_i, :, :] for logits_i, idx_i in zip(logits, idx)]).squeeze()
            batch_metrics = calculate_metrics_from_logits(logits, masks)

            bs = imgs.size(0)
            num_images += bs
            for k in metric_sums.keys():
                metric_sums[k] += batch_metrics[k] * bs  # weight by number of images

    return {k: (metric_sums[k] / max(1, num_images)) for k in metric_sums.keys()}


# --------------------
# Dataset + Dataloaders
# --------------------
train_ds, val_ds, test_ds = load_echodyna_downstream_datasets(allow_missing_masks=False)

def collate_fn(batch, frames):
    # fixed clip length across batch
    L = min(min(int(x["video"].shape[1]) for x in batch), int(frames))
    if L <= 0:
        raise RuntimeError("Invalid clip length computed (L <= 0).")

    kept = []  # (clip, mask[1,H,W], key_in_clip)
    for b, x in enumerate(batch):
        v = x["video"]  # [C,T,H,W]
        if not x.get("has_masks", False):
            raise RuntimeError("Segmentation collate_fn requires has_masks=True for all items.")

        fi = torch.as_tensor(x["frame_indices"], dtype=torch.long, device=v.device)
        m = x["masks"]
        m = m if torch.is_tensor(m) else torch.as_tensor(m)
        if fi.numel() != 2 or m.ndim != 3 or m.shape[0] != 2:
            raise RuntimeError(f"Expected 2 keyframes and masks [2,H,W], got fi={fi.numel()} m={tuple(m.shape)} for item {b}.")

        m = (m.to(v.device, non_blocking=True) > 0).float()  # [2,H,W]
        T = v.shape[1]

        for k in (0, 1):
            f = int(fi[k])
            if not (0 <= f < T):
                raise RuntimeError(f"Keyframe {f} out of bounds for T={T} (item {b}).")

            lo, hi = max(0, f - L + 1), min(f, T - L)
            if lo > hi:  # extra safety
                continue

            s = int(torch.randint(lo, hi + 1, (1,), device=v.device))
            clip, mask = v[:, s:s + L], m[k].unsqueeze(0)

            kept.append((clip, mask, f - s))

    if not kept:
        raise RuntimeError(f"All samples were skipped (L={L}, frames={frames}).")

    imgs, masks, keyframe_idx = zip(*kept)
    imgs = torch.stack(imgs)                       # [2*B,C,L,H,W]
    masks = torch.stack(masks)                     # [2*B,1,H,W]
    keyframe_idx = torch.tensor(keyframe_idx, device=imgs.device, dtype=torch.long)

    return {"imgs": imgs, "masks": masks, "keyframe_idx": keyframe_idx}



def train_collate_fn(batch):
    return collate_fn(batch, frames)

def val_collate_fn(batch):
    return collate_fn(batch, frames)

train_dl = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn,
    num_workers=16, pin_memory=True)
val_dl = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_collate_fn,
    num_workers=16)
test_dl = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=val_collate_fn,
    num_workers=16)


# --------------------
# Losses
# --------------------
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
        idx = batch["keyframe_idx"]

        optimizer.zero_grad(set_to_none=True)
        imgs = augmentations(imgs.transpose(1, 2)).transpose(1, 2).contiguous()
        logits = model(imgs)
        logits = torch.stack([logits_i[:, idx_i, :, :] for logits_i, idx_i in zip(logits, idx)])
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
            idx = batch["keyframe_idx"]

            logits = model(imgs)
            logits = torch.stack([logits_i[:, idx_i, :, :] for logits_i, idx_i in zip(logits, idx)])
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