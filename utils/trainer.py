import os
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---- plots ----
def plot_recons(mae, val_ds, output_dir, device, n_plot=8, fname="recons.png"):
    mae.eval()

    vid_idx = int(torch.randint(len(val_ds), (1,)).item())
    sample = val_ds[vid_idx]

    video = sample["video"]            # [T, C, H, W]
    video = video * 2 - 1              # [0,1] → [-1,1]
    timestamps = sample["timestamps"]  # [T]

    with torch.no_grad():
        video_b = video.unsqueeze(0).to(device)         # [1, T, C, H, W]
        ts_in = timestamps.unsqueeze(0).to(device)      # [1, T]
        out = mae(video_b, ts_in, return_pred=True)

        recon_mae = out["pred"].squeeze(0).detach().cpu()        # [T, C, H, W]
        recon_frames = out["pred_frames"].squeeze(0).detach().cpu()

    video = video.cpu()
    T = video.shape[0]

    # Randomly sample frame indices for plotting
    n_plot = min(n_plot, T)
    frame_idxs = torch.randperm(T)[:n_plot]
    frame_idxs, _ = torch.sort(frame_idxs)  # keep chronological order for display

    video_p = video[frame_idxs]
    mae_p = recon_mae[frame_idxs]
    frames_p = recon_frames[frame_idxs]

    def to_numpy(img_t):
        img_t = (img_t + 1.0) / 2.0  # [-1,1] → [0,1]
        arr = img_t.permute(1, 2, 0).numpy()
        arr = arr.clip(0.0, 1.0)
        return arr[:, :, 0] if arr.shape[2] == 1 else arr

    fig, axs = plt.subplots(3, n_plot, figsize=(n_plot * 2, 6))

    for i in range(n_plot):
        orig_np = to_numpy(video_p[i])
        mae_np = to_numpy(mae_p[i])
        frames_np = to_numpy(frames_p[i])

        for r in range(3):
            axs[r, i].axis("off")

        if orig_np.ndim == 2:
            axs[0, i].imshow(orig_np, cmap="gray")
            axs[1, i].imshow(mae_np, cmap="gray")
            axs[2, i].imshow(frames_np, cmap="gray")
        else:
            axs[0, i].imshow(orig_np)
            axs[1, i].imshow(mae_np)
            axs[2, i].imshow(frames_np)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
    plt.close(fig)


# ---- trainer ----
@dataclass
class TrainerConfig:
    output_dir: str
    epochs: int
    autocast: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    torch_compile: bool = False
    grad_clip_max_norm: float = 1.0
    save_every_epoch: bool = True


class MAETrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        train_dl,
        val_dl,
        val_ds=None,
        augmentations: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        config: Optional[TrainerConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.val_ds = val_ds
        self.augmentations = augmentations
        self.cfg = config or TrainerConfig(output_dir="outputs", epochs=1)

        os.makedirs(self.cfg.output_dir, exist_ok=True)

        # Keep total loss if you still want it for debugging / reference
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # NEW: per-component losses for plotting
        self.train_loss_mae: List[float] = []
        self.train_loss_frame: List[float] = []
        self.val_loss_mae: List[float] = []
        self.val_loss_frame: List[float] = []

        if self.cfg.torch_compile:
            self.model = torch.compile(self.model)

    def _save_checkpoint(self, name: str = "VMAE.pth"):
        path = os.path.join(self.cfg.output_dir, name)
        if self.cfg.torch_compile:
            torch.save(self.model._orig_mod.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def _save_loss_plot(self, run_val=True):
        fig, ax = plt.subplots(figsize=(8, 6))

        epochs = range(1, len(self.train_loss_mae) + 1)

        ax.plot(epochs, self.train_loss_mae, label="Train MAE")
        ax.plot(epochs, self.train_loss_frame, label="Train Frame")

        if run_val and len(self.val_loss_mae) > 0:
            ax.plot(epochs, self.val_loss_mae, label="Val MAE")
            ax.plot(epochs, self.val_loss_frame, label="Val Frame")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("MAE and Frame Loss over Epochs")

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "losses.png"), bbox_inches="tight")
        plt.close(fig)

    def _forward(self, videos, timestamps, target=None) -> Dict[str, Any]:
        """
        videos: [B, T, C, H, W]
        target: [B, T, C, H, W] or None
        """
        with torch.autocast(
            device_type="cuda",
            dtype=self.cfg.amp_dtype,
            enabled=(self.cfg.autocast and self.device.type == "cuda"),
        ):
            if target is None:
                out = self.model(videos, timestamps, return_pred=False)
            else:
                out = self.model(videos, timestamps, target=target, return_pred=False)
            return out

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        running_total = 0.0
        running_mae = 0.0
        running_frame = 0.0

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch_idx+1}/{self.cfg.epochs}")
        for batch in pbar:
            videos = batch["video"].to(self.device, non_blocking=True)  # [B,T,C,H,W]
            if self.augmentations:
                aug_videos = batch["aug_video"].to(self.device, non_blocking=True)  # [B,T,C,H,W]
            else:
                aug_videos = videos
            timestamps = batch["timestamps"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            out = self._forward(aug_videos, timestamps, target=videos)  # Reconstruct original videos

            loss = out["loss"]
            loss.backward()

            norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip_max_norm)
            self.optimizer.step()

            bsz = videos.size(0)
            running_total += loss.item() * bsz
            running_mae += out["loss_mae"].item() * bsz
            running_frame += out["loss_frame"].item() * bsz

            pbar.set_postfix(
                {"MAE": out["loss_mae"].item(), "Motion": out["loss_frame"].item(), "Grad Norm": float(norm)}
            )

        n = len(self.train_dl.dataset)
        return running_total / n, running_mae / n, running_frame / n

    @torch.no_grad()
    def validate_one_epoch(self, epoch_idx: int):
        self.model.eval()
        running_total = 0.0
        running_mae = 0.0
        running_frame = 0.0

        pbar = tqdm(self.val_dl, desc=f"Validation Epoch {epoch_idx+1}/{self.cfg.epochs}")
        for batch in pbar:
            videos = batch["video"].to(self.device, non_blocking=True)  # [B,T,C,H,W]
            timestamps = batch["timestamps"].to(self.device, non_blocking=True)

            out = self._forward(videos, timestamps, target=None)

            bsz = videos.size(0)
            running_total += out["loss"].item() * bsz
            running_mae += out["loss_mae"].item() * bsz
            running_frame += out["loss_frame"].item() * bsz

            pbar.set_postfix({"MAE": out["loss_mae"].item(), "Motion": out["loss_frame"].item()})

        n = len(self.val_dl.dataset)
        return running_total / n, running_mae / n, running_frame / n

    def train(self, run_val=True):
        for epoch in range(self.cfg.epochs):
            train_total, train_mae, train_frame = self.train_one_epoch(epoch)
            self.train_losses.append(train_total)
            self.train_loss_mae.append(train_mae)
            self.train_loss_frame.append(train_frame)

            if run_val:
                val_total, val_mae, val_frame = self.validate_one_epoch(epoch)
                self.val_losses.append(val_total)
                self.val_loss_mae.append(val_mae)
                self.val_loss_frame.append(val_frame)

            if self.scheduler is not None:
                self.scheduler.step()

            if run_val:
                print(
                    f"Epoch [{epoch+1}/{self.cfg.epochs}], "
                    f"Train MAE: {train_mae:.4f}, Train Frame: {train_frame:.4f}, "
                    f"Val MAE: {val_mae:.4f}, Val Frame: {val_frame:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{self.cfg.epochs}], "
                    f"Train MAE: {train_mae:.4f}, Train Frame: {train_frame:.4f}"
                )

            if self.cfg.save_every_epoch:
                self._save_checkpoint("VMAE.pth")
                if self.val_ds is not None:
                    plot_recons(self.model, self.val_ds, self.cfg.output_dir, self.device)
                self._save_loss_plot(run_val=run_val)

        return {
            "train_loss_mae": self.train_loss_mae,
            "train_loss_frame": self.train_loss_frame,
            "val_loss_mae": self.val_loss_mae,
            "val_loss_frame": self.val_loss_frame,
            # totals kept (optional)
            "train_total": self.train_losses,
            "val_total": self.val_losses,
        }
