import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from models.VideoViT import VideoViTEncoder, VideoViTCfg
from models.Downstream import LV_Segmentation
from datahandling.collate import LV_collate
from datahandling.augmentations.get_augmentations import get_pretrain_augmentations


# --------------------
# Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dir = "results/2026_02_15/15_28_VMAE"
ckpt_name = "VMAE.pth"
config = json.load(open("config/VMAE.json", "r"))
output_dir = os.path.join(load_dir, "LVSeg")
os.makedirs(output_dir, exist_ok=True)

# Training Parameters
epochs = 60
batch_size = 16
lr = 3e-4
weight_decay = 1e-3
dropout = 0.1
frames = config["training"]["max_frames"]
torch.set_float32_matmul_precision('high')
torch_compile = False

train_params = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "dropout": dropout,
    "frames": frames,
}

comments = [
    ""
]


enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
pretrained_dict = torch.load(os.path.join(load_dir, "VMAE.pth"), map_location=device)
enc.load_state_dict({k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if k.startswith("encoder.")})
model = LV_Segmentation(encoder=enc, dropout=dropout).to(device)
print(f"Initialized LV Segmentation with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e3:.2f}K trainable parameters.")
model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
if torch_compile:
    model = torch.compile(model)
autocast = config["training"]["autocast"]

enc = list(model.encoder.parameters())
optimizer = torch.optim.AdamW([
    {"params": enc, "lr": lr/4, "weight_decay": weight_decay/4},
    {"params": [p for p in model.parameters() if id(p) not in {id(p) for p in enc}],
     "lr": lr, "weight_decay": weight_decay}])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# --------------------
# Metrics
# --------------------
def _safe_div_tensor(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return num / (den + eps)

@torch.inference_mode()
def calculate_metrics_from_logits(logits: torch.Tensor, masks: torch.Tensor):
    # logits: [B,1,H,W], masks: [B,1,H,W]
    preds = (logits >= 0).to(torch.int64)
    targets = (masks >= 0.5).to(torch.int64)

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
# Bugged
def save_examples_echo_dyna(model, val_ds, out_dir, results=5):
    results = min(results, len(val_ds))
    random_indices = random.sample(range(len(val_ds)), results)

    fig, axes = plt.subplots(results, 3, figsize=(18, 6 * results))
    if results == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    for i, idx in enumerate(random_indices):
        item = val_ds[idx]
        video = item["video"].unsqueeze(0).to(device)  # [1,T,C,H,W]
        timestamps = item["timestamps"].unsqueeze(0).to(device)  # [1,T]
        frames_idx = item["masks"]["frame_indices"]   # 2 keyframes
        masks = item["masks"]["masks"]                # [2,H,W] (or tensor)
        frame = int(frames_idx[0])

        true_mask = masks[0]
        true_mask = true_mask if torch.is_tensor(true_mask) else torch.as_tensor(true_mask)
        true_mask = true_mask.float().cpu().numpy()

        with torch.inference_mode():
            logits = model(video, timestamps, autocast=autocast)  # [1,1,H,W]
            pred = torch.sigmoid(logits)[0, frame].cpu().numpy()  # [H,W], keyframe prediction
            img = video[0, frame]  # [C,H,W]


        # display image
        img_np = img[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
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
        overlay[:, :, 1] = (((pred > 0.5) & (true_mask == 1)) * 255).astype(np.uint8)  # TP green
        overlay[:, :, 0] = (((pred > 0.5) & (true_mask == 0)) * 255).astype(np.uint8)  # FP red
        overlay[:, :, 0] += (((pred <= 0.5) & (true_mask == 1)) * 255).astype(np.uint8)  # FN amber
        overlay[:, :, 1] += (((pred <= 0.5) & (true_mask == 1)) * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay)

        axes[i, 0].imshow(image_pil)
        axes[i, 0].set_title("Original Image", fontsize=18)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(predicted_pil, cmap="gray")
        axes[i, 1].set_title("Predicted Probabilities", fontsize=18)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay_pil)
        axes[i, 2].set_title("Overlay (G:TP, Y:FN, R:FP)", fontsize=18)
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
        f.write(f"Model size (trainable): {model_size/1e6:.3f} M\n")
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
# Dataset + Dataloaders
# --------------------
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=True)

def train_collate_fn(batch):
    return LV_collate(batch, max_frames=frames, augmentations=get_pretrain_augmentations())

def val_collate_fn(batch):
    return LV_collate(batch, max_frames=frames)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn,
                      num_workers=24, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_collate_fn,
                    num_workers=16, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=val_collate_fn,
                     num_workers=16, pin_memory=True)


# --------------------
# Losses
# --------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        B = logits.size(0)
        probs = torch.sigmoid(logits).view(B, -1)
        targets = targets.view(B, -1)
        inter = (probs * targets).sum(dim=-1)
        total = probs.sum(dim=-1) + targets.sum(dim=-1)
        return (1 - (2. * inter + self.smooth) / (total + self.smooth)).mean()

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

        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
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
# Helpers: pick keyframe logits from [B,T,1,H,W] -> [B,1,H,W]
# --------------------
def gather_keyframe_logits(video_logits: torch.Tensor, key_idx: torch.Tensor) -> torch.Tensor:
    # video_logits: [B,T,1,H,W], key_idx: [B]
    B, T, C, H, W = video_logits.shape
    out = torch.stack([video_logits[b, key_idx[b], 0] for b in range(B)], dim=0)  # [B,H,W]
    return out.unsqueeze(2)  # [B,2,1,H,W]


# --------------------
# Test
# --------------------
@torch.inference_mode()
def test_model(model, test_dl):
    model.eval()
    metric_sums = {"jaccard": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "dice": 0.0}
    num_images = 0

    for batch in tqdm(test_dl, desc="Testing"):
        videos = batch["video"].to(device, non_blocking=True)          # [B,T,C,H,W]
        timestamps = batch["timestamps"].to(device, non_blocking=True)  # [B
        masks = batch["masks"].to(device, non_blocking=True)   # [B,1,H,W]
        idx = batch["frame_indices"].to(device)

        logits_vid = model(videos, timestamps, autocast=autocast)  # [B,T,1,H,W]
        logits = gather_keyframe_logits(logits_vid, idx)       # [B,1,H,W]
        batch_metrics = calculate_metrics_from_logits(logits, masks)

        bs = videos.size(0)
        num_images += bs
        for k in metric_sums.keys():
            metric_sums[k] += batch_metrics[k] * bs

    return {k: (metric_sums[k] / max(1, num_images)) for k in metric_sums.keys()}


# --------------------
# Training
# --------------------
train_losses = []
val_losses = []
val_metrics = {"jaccard": [], "precision": [], "recall": [], "accuracy": [], "dice": []}

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
        video = batch["video"].to(device)      # [B,T,C,H,W]
        timestamps = batch["timestamps"].to(device)  # [B,T]
        masks = batch["masks"].to(device)    # [B,1,H,W]
        idx = batch["frame_indices"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits_vid = model(video, timestamps, autocast=autocast)  # [B,T,1,H,W]
        logits = gather_keyframe_logits(logits_vid, idx)         # [B,1,H,W]

        focal = focal_loss(logits, masks)
        dloss = dice_loss(logits, masks)
        loss = focal + dloss

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = video.size(0)
        train_loss_sum += loss.item() * bs
        num_train_samples += bs

        p_bar.set_postfix({"Focal": focal.item(), "DiceLoss": dloss.item(), "GradNorm": float(norm)})

    train_loss = train_loss_sum / max(1, num_train_samples)
    train_losses.append(train_loss)

    # ---- Validate ----
    model.eval()
    val_loss_sum = 0.0
    num_val_samples = 0

    metric_sums = {"jaccard": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "dice": 0.0}
    num_val_images = 0

    with torch.inference_mode():
        p_bar = tqdm(val_dl, desc=f"Validation {epoch+1}/{epochs}")
        for batch in p_bar:
            video = batch["video"].to(device)
            timestamps = batch["timestamps"].to(device)
            masks = batch["masks"].to(device)
            idx = batch["frame_indices"].to(device)

            logits_vid = model(video, timestamps, autocast=autocast)  # [B,T,1,H,W]
            logits = gather_keyframe_logits(logits_vid, idx)

            focal = focal_loss(logits, masks)
            dloss = dice_loss(logits, masks)
            loss = focal + dloss

            bs = video.size(0)
            val_loss_sum += loss.item() * bs
            num_val_samples += bs

            batch_metrics = calculate_metrics_from_logits(logits, masks)
            num_val_images += bs
            for k in metric_sums.keys():
                metric_sums[k] += batch_metrics[k] * bs

            p_bar.set_postfix({"Focal": focal.item(), "DiceLoss": dloss.item()})

    val_loss = val_loss_sum / max(1, num_val_samples)
    val_losses.append(val_loss)

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
    results = {"train_losses": train_losses, "val_losses": val_losses, "val_metrics": val_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # ---- Plots + qualitative examples ----
    # save_examples_echo_dyna(model, val_ds, output_dir, results=5)
    plot_losses_and_dice(train_losses, val_losses, val_metrics["dice"], output_dir)

    # ---- results.txt ----
    write_results_txt(
        output_dir,
        model=model,
        model_size=model_size,
        train_params=train_params,
        comments=comments,
        best_results=best_results,
        start_time=start_time,
    )

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
    test_results=test_metrics,
)
