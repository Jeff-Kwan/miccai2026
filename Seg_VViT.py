import os
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
from models.VideoViT import VideoViTEncoder, VideoViTCfg  # encoder backbone


# --------------------
# Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dir = "results/2026_02_09/17_02_VMAE"  # <-- change to your VMAE pretrain run
ckpt_name = "VMAE.pth"                      # <-- change if needed
output_dir = os.path.join(load_dir, "LVSeg")
os.makedirs(output_dir, exist_ok=True)

# Training Parameters
epochs = 100
batch_size = 16
learning_rate = 3e-4
weight_decay = 1e-2
frames = 32

train_params = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "frames": frames,
}

comments = [
    "VideoViTEncoder backbone from VideoViTMAE pretraining",
    "Seg head: conv -> upsample to pixel logits",
    "Keyframe-supervised (two keyframes per video)",
]


# --------------------
# Augmentations (expects [B,T,C,H,W], values in [-1,1])
# --------------------
class RandomGamma(nn.Module):
    def __init__(self, gamma=(0.7, 1.5)):
        super().__init__()
        self.gamma = gamma

    def forward(self, x):
        x = (x + 1) * 0.5
        g = torch.empty(1, device=x.device).uniform_(*self.gamma)
        return (x.pow(g) * 2 - 1).clamp(-1.0, 1.0)

class ClipBrightnessContrast(nn.Module):
    def __init__(self, brightness=0.3, contrast=0.2):
        super().__init__()
        self.b = brightness
        self.c = contrast

    def forward(self, x):
        b = torch.empty(1, device=x.device).uniform_(-self.b, self.b)
        c = torch.empty(1, device=x.device).uniform_(1 - self.c, 1 + self.c)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return ((x - mean) * c + mean + b).clamp(-1.0, 1.0)

class SpeckleNoise(nn.Module):
    def __init__(self, std=(0.02, 0.1)):
        super().__init__()
        self.std = std

    def forward(self, x):
        x = (x + 1) * 0.5
        std = torch.empty(1, device=x.device).uniform_(*self.std)
        noise = torch.randn_like(x) * std
        x = (x + x * noise).clamp(0.0, 1.0)
        return x * 2 - 1

augmentations = v2.Compose([
    v2.RandomApply([
        v2.RandomChoice([
            v2.RandomChoice([
                ClipBrightnessContrast(brightness=0.3, contrast=0.2),
                RandomGamma(gamma=(0.7, 1.5)),
            ]),
            v2.RandomChoice([
                v2.RandomAdjustSharpness(sharpness_factor=0.5, p=1),
                v2.GaussianBlur(kernel_size=7, sigma=(0.25, 1.5)),
            ]),
            v2.RandomChoice([
                v2.GaussianNoise(0, 0.05),
                SpeckleNoise(std=(0.02, 0.1)),
            ]),
        ])
    ], p=0.5),
    v2.RandomApply([v2.RandomErasing(p=1)], p=0.3),
])


# --------------------
# Model: VideoViT encoder + segmentation head
# --------------------
class VideoViTSegHead(nn.Module):
    """
    Uses VideoViTEncoder patch tokens (grid h x w) and upsamples to pixel logits.

    Expects encoder to return enc_tokens shaped [B,T,N,D] where N=h*w.
    Produces logits [B,T,1,H,W].
    """
    def __init__(self, encoder: nn.Module, patch: int, enc_dim: int):
        super().__init__()
        self.encoder = encoder
        self.patch = int(patch)
        self.enc_dim = int(enc_dim)

        mid = max(64, enc_dim // 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(enc_dim, enc_dim, 3, 1, 1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(enc_dim, mid, 3, 1, 1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(mid, mid//2, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(mid//2, 2, 2, 2, 0),
            nn.Conv2d(2, 1, 1, 1, 0))
        

    @staticmethod
    def _all_keep_idx(B: int, T: int, N: int, device) -> torch.Tensor:
        base = torch.arange(N, device=device, dtype=torch.long)
        return base.view(1, 1, N).expand(B, T, N).contiguous()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: [B,T,C,H,W]
        B, T, C, H, W = video.shape
        if H % self.patch != 0 or W % self.patch != 0:
            raise ValueError(f"H,W must be divisible by patch={self.patch}, got {(H,W)}")

        h = H // self.patch
        w = W // self.patch
        N = h * w

        keep_idx = self._all_keep_idx(B, T, N, video.device)

        # encoder is assumed to accept keep_idx and return (gcls, enc_tokens, hw)
        with torch.no_grad():
            gcls, enc_tokens, hw_enc = self.encoder(video, keep_idx=keep_idx)

        enc_tokens = enc_tokens[:, :, 1:, :]    # Remove CLS token
        if hw_enc != (h, w):
            raise ValueError(f"Encoder hw={hw_enc} != expected {(h,w)} from input {(H,W)} patch={self.patch}")

        # enc_tokens: [B,T,N,D] -> [B*T, D, h, w]
        if enc_tokens.dim() != 4 or enc_tokens.shape[:3] != (B, T, N):
            raise RuntimeError(f"Unexpected enc_tokens shape {tuple(enc_tokens.shape)}; expected {(B,T,N,'D')}")

        D = enc_tokens.size(-1)
        x = enc_tokens.view(B * T, N, D).transpose(1, 2).contiguous()  # [BT,D,N]
        x = x.view(B * T, D, h, w)                                     # [BT,D,h,w]
        x = self.fuse(x)                                               # [BT,mid,h,w]
        x = x.view(B, T, 1, H, W)
        return x


# Build encoder to match your pretrain config
# (This should match how you built enc in your VMAE pretrain script.)
enc_cfg = VideoViTCfg(dim=384, depth=8, heads=6, patch=8)
encoder = VideoViTEncoder(enc_cfg)

model = VideoViTSegHead(encoder=encoder, patch=enc_cfg.patch, enc_dim=enc_cfg.dim).to(device)

# Load VMAE checkpoint (partial load is fine)
ckpt_path = os.path.join(load_dir, ckpt_name)
pretrained = torch.load(ckpt_path, map_location=device)

model_dict = model.state_dict()
matched = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(matched)
missing, unexpected = model.load_state_dict(model_dict, strict=False)

print(f"Loaded pretrained weights from: {ckpt_path}")
print(f"Matched keys: {len(matched)} | Missing: {len(missing)} | Unexpected: {len(unexpected)}")

# Freeze encoder if you want “linear probe / head-only” finetune.
# If you want full finetune, comment these lines.
model.encoder.requires_grad_(False)

model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Initialized VideoViT-Seg with {model_size/1e6:.2f}M trainable parameters.")

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=learning_rate, weight_decay=weight_decay)
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
def save_examples_echo_dyna(model, val_ds, out_dir, results=5):
    results = min(results, len(val_ds))
    random_indices = random.sample(range(len(val_ds)), results)

    fig, axes = plt.subplots(results, 3, figsize=(18, 6 * results))
    if results == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    for i, idx in enumerate(random_indices):
        item = val_ds[idx]
        frames_idx = item["frame_indices"]   # 2 keyframes
        masks = item["masks"]                # [2,H,W] (or tensor)
        frame = int(frames_idx[0])

        true_mask = masks[0]
        true_mask = true_mask if torch.is_tensor(true_mask) else torch.as_tensor(true_mask)
        true_mask = true_mask.float().cpu().numpy()

        # single-frame clip with T=1 (still in [T,C,H,W] convention for model input)
        # dataset video is [C,T,H,W] -> take one frame -> [C,H,W]
        img = item["video"][:, frame].unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]
        img = img.to(device)

        with torch.inference_mode():
            logits = model(img)                 # [1,1,1,H,W]
            pred = torch.sigmoid(logits)[0, 0, 0].cpu().numpy()

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

def collate_fn(batch, frames):
    # fixed clip length across batch
    L = min(min(int(x["video"].shape[1]) for x in batch), int(frames))
    if L <= 0:
        raise RuntimeError("Invalid clip length computed (L <= 0).")

    kept = []  # (clip[T,C,H,W], mask[1,H,W], key_in_clip)
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
            if lo > hi:
                continue

            s = int(torch.randint(lo, hi + 1, (1,), device=v.device))
            clip = v[:, s:s + L]                    # [C,L,H,W]
            clip = clip.permute(1, 0, 2, 3)         # [L,C,H,W] == [T,C,H,W]
            mask = m[k].unsqueeze(0)                # [1,H,W]
            kept.append((clip, mask, f - s))

    if not kept:
        raise RuntimeError(f"All samples were skipped (L={L}, frames={frames}).")

    imgs, masks, keyframe_idx = zip(*kept)
    imgs = torch.stack(imgs)                       # [2*B,T,C,H,W]
    masks = torch.stack(masks)                     # [2*B,1,H,W]
    keyframe_idx = torch.tensor(keyframe_idx, device=imgs.device, dtype=torch.long)

    return {"imgs": imgs, "masks": masks, "keyframe_idx": keyframe_idx}

def train_collate_fn(batch):
    return collate_fn(batch, frames)

def val_collate_fn(batch):
    return collate_fn(batch, frames)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn,
                      num_workers=16, pin_memory=True)
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
    if C != 1:
        raise ValueError(f"Expected 1 channel logits, got C={C}")
    out = torch.stack([video_logits[b, int(key_idx[b]), 0] for b in range(B)], dim=0)  # [B,H,W]
    return out.unsqueeze(1)  # [B,1,H,W]


# --------------------
# Test
# --------------------
@torch.inference_mode()
def test_model(model, test_dl):
    model.eval()
    metric_sums = {"jaccard": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "dice": 0.0}
    num_images = 0

    for batch in tqdm(test_dl, desc="Testing"):
        imgs = batch["imgs"].to(device, non_blocking=True)     # [B,T,C,H,W]
        masks = batch["masks"].to(device, non_blocking=True)   # [B,1,H,W]
        idx = batch["keyframe_idx"].to(device)

        logits_vid = model(imgs)                               # [B,T,1,H,W]
        logits = gather_keyframe_logits(logits_vid, idx)       # [B,1,H,W]
        batch_metrics = calculate_metrics_from_logits(logits, masks)

        bs = imgs.size(0)
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
        imgs = batch["imgs"].to(device)      # [B,T,C,H,W]
        masks = batch["masks"].to(device)    # [B,1,H,W]
        idx = batch["keyframe_idx"].to(device)

        optimizer.zero_grad(set_to_none=True)

        imgs_aug = augmentations(imgs)
        logits_vid = model(imgs_aug)                            # [B,T,1,H,W]
        logits = gather_keyframe_logits(logits_vid, idx)         # [B,1,H,W]

        focal = focal_loss(logits, masks)
        dloss = dice_loss(logits, masks)
        loss = focal + dloss

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = imgs.size(0)
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
            imgs = batch["imgs"].to(device)
            masks = batch["masks"].to(device)
            idx = batch["keyframe_idx"].to(device)

            logits_vid = model(imgs)
            logits = gather_keyframe_logits(logits_vid, idx)

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
    save_examples_echo_dyna(model, val_ds, output_dir, results=5)
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
