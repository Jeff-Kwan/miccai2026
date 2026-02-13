import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class TrainerConfig:
    output_dir: str
    epochs: int
    autocast: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    torch_compile: bool = False
    grad_clip_max_norm: float = 1.0
    save_every_epoch: bool = True


class PreTrainer:
    def __init__(
        self,
        paradigm: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        train_dl,
        augmented_input: bool,
        config: Optional[TrainerConfig] = None,
    ):
        self.paradigm = paradigm.strip().lower()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_dl = train_dl
        self.augmented_input = augmented_input
        self.cfg = config or TrainerConfig(output_dir="outputs", epochs=1)

        os.makedirs(self.cfg.output_dir, exist_ok=True)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        if self.cfg.torch_compile:
            self.model = torch.compile(self.model)

    def _model_for_saving(self) -> nn.Module:
        return self.model._orig_mod if self.cfg.torch_compile else self.model

    def _save_checkpoint(self, name: str = "VJEPA.pth"):
        path = os.path.join(self.cfg.output_dir, name)
        torch.save(self._model_for_saving().state_dict(), path)

    def _save_loss_plot(self):
        epochs = range(1, len(self.train_losses) + 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, self.train_losses, label="Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Loss over Epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "losses.png"), bbox_inches="tight")
        plt.close(fig)

    def _forward(self, videos, timestamps, target=None) -> Dict[str, Any]:
        """
        videos: [B, T, C, H, W]  (student input; may be augmented)
        target: [B, T, C, H, W] or None (aux frame recon target; often original)
        """
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.cfg.amp_dtype,
            enabled=(self.cfg.autocast and self.device.type == "cuda"),
        ):
            if target is None:
                out = self.model(videos, timestamps)
            else:
                out = self.model(videos, timestamps, target=target)
        return out

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()

        running_total = 0.0

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch_idx+1}/{self.cfg.epochs}")
        for batch in pbar:
            videos = batch["video"].to(self.device, non_blocking=True)          # [B,T,C,H,W] original
            timestamps = batch["timestamps"].to(self.device, non_blocking=True) # [B,T]

            # student input
            if self.augmented_input:
                aug_videos = batch["aug_video"].to(self.device, non_blocking=True)
            else:
                aug_videos = videos

            self.optimizer.zero_grad(set_to_none=True)

            # Student sees aug_videos; aux frame loss reconstructs videos
            out = self._forward(aug_videos, timestamps, target=videos)

            loss = out["loss"]
            loss.backward()

            norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip_max_norm)
            self.optimizer.step()

            if self.paradigm == 'jepa':
                # EMA update
                self._model_for_saving().update_ema()

            bsz = videos.size(0)
            running_total += loss.item() * bsz

            pbar.set_postfix(
                {
                    "Loss": float(out["loss"].item()),
                    "GradNorm": float(norm),
                }
            )

        n = len(self.train_dl.dataset)
        return running_total / n

    def train(self):
        for epoch in range(self.cfg.epochs):
            train_total = self.train_one_epoch(epoch)
            self.train_losses.append(train_total)

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch [{epoch+1}/{self.cfg.epochs}] - Train Loss: {train_total:.4f}")

            if self.cfg.save_every_epoch:
                self._save_loss_plot()
                self._save_checkpoint("VJEPA.pth")

        return {
            "train_total": self.train_losses,
        }