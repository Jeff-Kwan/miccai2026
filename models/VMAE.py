import torch
import torch.nn as nn
import torch.nn.functional as F

from VideoViT import VideoViTEncoder, VideoViTDecoder


class VideoMAE(nn.Module):
    """
    Video MAE (no global CLS).

    - keep_idx selects visible patches (constant Nvis across frames)
    - encoder encodes only visible patches and returns:
        frame_cls:    (B,T,D)
        frame_tokens: (B,T,Nvis,D)
    - decoder reconstructs masked patches only (B,T,Nmask,patch_dim)
    - loss computed on masked patches only

    NOTE (w/ temporal RoPE):
      - encoder.forward requires timestamps: (B,T)
      - decoder.forward requires timestamps: (B,T)
    """
    def __init__(
        self,
        encoder: VideoViTEncoder,
        decoder: VideoViTDecoder,
        *,
        norm_pix_loss: bool = False,
        loss_type: str = "mse",     # "mse" | "l1" | "smooth_l1"
        mask_ratio: float = 0.75,   # used only if keep_idx not provided
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.norm_pix_loss = bool(norm_pix_loss)
        self.loss_type = loss_type
        self.mask_ratio = float(mask_ratio)

        # Convenience: pull patch/in_chans from decoder if present
        self.patch = getattr(decoder, "patch", None)
        self.in_chans = getattr(decoder, "in_chans", None)

    # --------------------- MAE helpers ---------------------

    @staticmethod
    def _patchify(x: torch.Tensor, patch: int) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        x: (B,T,C,H,W) -> (B,T,N,patch_dim), hw=(h,w)
        patch_dim = patch*patch*C
        """
        B, T, C, H, W = x.shape
        if H % patch != 0 or W % patch != 0:
            raise ValueError(f"H,W must be divisible by patch={patch}, got {(H, W)}")
        h, w = H // patch, W // patch
        x = x.view(B, T, C, h, patch, w, patch)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        x = x.view(B, T, h * w, patch * patch * C)
        return x, (h, w)

    @staticmethod
    def _random_keep_idx(B: int, T: int, N: int, Nvis: int, device) -> torch.Tensor:
        """
        Sample visible patches once per video (per b), then repeat across time.
        Output: (B,T,Nvis) with identical keep indices for all t.
        """
        idx_b = torch.empty((B, Nvis), device=device, dtype=torch.long)
        for b in range(B):
            idx_b[b] = torch.randperm(N, device=device)[:Nvis]
        return idx_b.unsqueeze(1).expand(B, T, Nvis).contiguous()

    @staticmethod
    def _masked_idx_sorted(N: int, keep_bt: torch.Tensor) -> torch.Tensor:
        """
        keep_bt: (BT,Nvis)
        Returns mask_idx: (BT,Nmask) indices of masked patches in ascending order.
        """
        BT, _ = keep_bt.shape
        keep_bool = torch.zeros(BT, N, device=keep_bt.device, dtype=torch.bool)
        keep_bool.scatter_(1, keep_bt, True)
        mask_idx = (~keep_bool).nonzero(as_tuple=False)[:, 1]
        return mask_idx.view(BT, N - keep_bt.size(1))

    @staticmethod
    def _gather_bt(x_bt: torch.Tensor, idx_bt: torch.Tensor) -> torch.Tensor:
        """
        x_bt: (BT,N,D)
        idx_bt: (BT,K)
        -> (BT,K,D)
        """
        BT, _, D = x_bt.shape
        K = idx_bt.size(1)
        return torch.gather(x_bt, 1, idx_bt.unsqueeze(-1).expand(BT, K, D))

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, target, reduction="mean")
        if self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction="mean")
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction="mean")
        raise ValueError(f"loss_type must be mse|l1|smooth_l1, got {self.loss_type}")

    # ----------------------------------- forward -----------------------------------

    def forward(
        self,
        video: torch.Tensor,                      # (B,T,C,H,W)
        timestamps: torch.Tensor,                 # (B,T) REQUIRED for temporal RoPE
        target: torch.Tensor | None = None,       # (B,T,C,H,W) optional target for patch loss
        *,
        keep_idx: torch.Tensor | None = None,     # (B,T,Nvis)
        mask_ratio: float | None = None,          # overrides self.mask_ratio if keep_idx is None
        return_pred: bool = False,
    ):
        B, T, C, H, W = video.shape

        if timestamps.dim() != 2 or timestamps.size(0) != B or timestamps.size(1) != T:
            raise ValueError(f"timestamps must be (B,T)=({B},{T}), got {tuple(timestamps.shape)}")
        timestamps = timestamps.to(device=video.device)

        if self.patch is None or self.in_chans is None:
            raise ValueError("Decoder must have attributes .patch and .in_chans (as in provided VideoViTDecoder).")
        if C != self.in_chans:
            raise ValueError(f"video C={C} must match decoder in_chans={self.in_chans}")

        # --- targets for MAE patch loss ---
        if target is None:
            target_full, hw = self._patchify(video, self.patch)   # (B,T,N,patch_dim)
        else:
            target_full, hw = self._patchify(target, self.patch)

        h, w = hw
        N = target_full.size(2)
        patch_dim = target_full.size(3)

        # --- keep_idx ---
        if keep_idx is None:
            r = self.mask_ratio if mask_ratio is None else float(mask_ratio)
            if not (0.0 < r < 1.0):
                raise ValueError(f"mask_ratio must be in (0,1), got {r}")
            Nvis = max(1, int(round(N * (1.0 - r))))
            keep_idx = self._random_keep_idx(B, T, N, Nvis, video.device)
        else:
            if keep_idx.dim() != 3 or keep_idx.size(0) != B or keep_idx.size(1) != T:
                raise ValueError(f"keep_idx must be (B,T,Nvis)=({B},{T},*), got {tuple(keep_idx.shape)}")
            keep_idx = keep_idx.to(device=video.device, dtype=torch.long)
            Nvis = keep_idx.size(2)

        # --- encoder (no gcls) ---
        frame_cls, frame_tokens, hw_enc = self.encoder(video, keep_idx=keep_idx, timestamps=timestamps)
        if hw_enc != hw:
            raise ValueError(f"Encoder hw={hw_enc} != patchify hw={hw}")

        # --- decode masked patches only ---
        pred_masked = self.decoder(
            frame_cls=frame_cls,
            frame_tokens=frame_tokens,
            keep_idx=keep_idx,
            hw=hw,
        )  # (B,T,Nmask,patch_dim)

        # --- build masked targets in the same order as decoder ---
        Nmask = N - Nvis
        BT = B * T

        target_bt = target_full.view(BT, N, patch_dim)
        keep_bt = keep_idx.view(BT, Nvis)

        mask_idx = self._masked_idx_sorted(N, keep_bt)            # (BT,Nmask)
        target_masked_bt = self._gather_bt(target_bt, mask_idx)   # (BT,Nmask,patch_dim)
        target_masked = target_masked_bt.view(B, T, Nmask, patch_dim)

        if self.norm_pix_loss:
            mean = target_masked.mean(dim=-1, keepdim=True)
            var = target_masked.var(dim=-1, keepdim=True, unbiased=False)
            target_masked = (target_masked - mean) / (var + 1e-6).sqrt()

        loss = self._loss(pred_masked, target_masked)

        out = {"loss": loss}

        # Optional: return patch-assembled reconstruction (visible patches copied from target_full for viz)
        if return_pred:
            pred_masked_bt = pred_masked.view(BT, Nmask, patch_dim)

            pred_full_bt = target_bt.clone().to(pred_masked_bt.dtype)
            pred_full_bt.scatter_(
                1,
                mask_idx.unsqueeze(-1).expand(BT, Nmask, patch_dim),
                pred_masked_bt,
            )
            pred_full_patches = pred_full_bt.view(B, T, N, patch_dim)

            pred_video = pred_full_patches.view(B, T, h, w, self.patch, self.patch, C)
            pred_video = pred_video.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
            pred_video = pred_video.view(B, T, C, h * self.patch, w * self.patch)

            out["pred"] = pred_video  # (B,T,C,H,W)

        return out


# ----------------------------- example -----------------------------
if __name__ == "__main__":
    import json
    from VideoViT import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg
    config = json.load(open("config/VMAE.json", "r"))
    enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
    dec = VideoViTDecoder(enc_dim=config["encoder"]["dim"], patch=config["encoder"]["patch"], 
                        in_chans=config["encoder"]["in_chans"], cfg=VideoViTDecCfg(**config["decoder"]))
    mae = VideoMAE(enc, dec, norm_pix_loss=config["mae"]["norm_pix_loss"], mask_ratio=config["mae"]["mask_ratio"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae = mae.to(device)

    B = config["training"]["batch_size"]
    T = config["training"]["max_frames"]
    C, H, W = 3, 112, 112
    video = torch.randn(B, T, C, H, W, device=device)

    # Example timestamps: 0..T-1 for each sample (you can pass real timestamps here)
    timestamps = torch.arange(T, device=device).view(1, T).expand(B, T)

    # Profile memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None,
        ],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with torch.autocast('cuda', torch.bfloat16):
            out = mae(video, timestamps=timestamps, return_pred=True)
            output = out["pred"]
            loss = out["loss"]
        loss.backward()

    assert output.shape == video.shape, f"Expected output shape {video.shape}, got {output.shape}"
    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    print(
        "Total trainable parameters:",
        round(sum(p.numel() for p in mae.parameters() if p.requires_grad) / 1e6, 2),
        "M",
    )
