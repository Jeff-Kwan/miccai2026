import torch
import torch.nn as nn
import torch.nn.functional as F

from .VideoViT import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg


class SimpleConvDecoder(nn.Module):
    """
    Input:  z of shape [B, T, latent]  (CLS tokens per frame)
    Output: x of shape [B, T, out_dim, 128, 128]
    """
    def __init__(self, latent: int, out_dim: int = 3, base: int = 256):
        super().__init__()

        self.proj = nn.Conv2d(latent, base, kernel_size=1)

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),   # 1->2->4->...->128
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
            )

        self.decoder = nn.Sequential(
            up_block(base,     base),       # 1   -> 2
            up_block(base,     base // 2),  # 2   -> 4
            up_block(base // 2, base // 4), # 4   -> 8
            up_block(base // 4, base // 8), # 8   -> 16
            up_block(base // 8, base // 16),# 16  -> 32
            up_block(base // 16, base // 32),# 32 -> 64
            nn.Upsample(scale_factor=2, mode="nearest"),       # 64 -> 128
            nn.Conv2d(base // 32, out_dim, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, latent)
        B, T, latent = z.shape
        z = z.view(B * T, latent, 1, 1)     # (B*T, latent, 1, 1)
        x = self.proj(z)                    # (B*T, base, 1, 1)
        x = self.decoder(x)                 # (B*T, out_dim, 128, 128)
        return x.view(B, T, x.size(1), 128, 128)  # (B, T, out_dim, 128, 128)


class VideoMotionMAE(nn.Module):
    """
    Video MAE with two reconstruction pathways:

      (A) MAE patch pathway (same as your VideoViTMAE):
          - keep_idx selects visible patches (constant Nvis across frames)
          - encoder encodes only visible patches (no padding / no SDPA masks)
          - decoder reconstructs masked patches only
          - loss computed on masked patches only

      (B) CLS -> frame pathway (added):
          - take per-frame CLS tokens from encoder output
          - decode them into full frames using a frame_decoder (e.g. SimpleConvDecoder)
          - compute a frame reconstruction loss (e.g. mse/l1/smooth_l1) against the input frames

    Returns:
      - always: {"loss": total_loss, "loss_mae": ..., "loss_frame": ...}
      - optionally: predictions for patch-assembled video and/or cls-decoded video
    """
    def __init__(
        self,
        encoder: VideoViTEncoder,
        decoder: VideoViTDecoder,
        frame_decoder: nn.Module,                # expects (B,T,enc_dim)->(B,T,C,H,W)
        motion_dim: int = 2,                     # dimensionality of motion basis (default 2 for 2D translation)
        *,
        norm_pix_loss: bool = False,
        loss_type: str = "mse",                  # "mse" | "l1" | "smooth_l1"
        mask_ratio: float = 0.75,                # used only if keep_idx not provided
        frame_loss_weight: float = 1.0,
        mae_loss_weight: float = 1.0,
        frame_target_size: tuple[int, int] = (128, 128),  # matches SimpleConvDecoder
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.frame_decoder = frame_decoder

        self.norm_pix_loss = norm_pix_loss
        self.loss_type = loss_type
        self.mask_ratio = float(mask_ratio)

        self.frame_loss_weight = float(frame_loss_weight)
        self.mae_loss_weight = float(mae_loss_weight)
        self.frame_target_size = tuple(frame_target_size)

        if loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(f"loss_type must be mse|l1|smooth_l1, got {loss_type}")

        # Convenience: pull patch/in_chans from decoder if present
        self.patch = getattr(decoder, "patch", None)
        self.in_chans = getattr(decoder, "in_chans", None)

        # Motion!
        self.template_mlp = nn.Sequential(
            nn.Linear(encoder.cfg.dim, encoder.cfg.dim*2),
            nn.GELU(),
            nn.Linear(encoder.cfg.dim*2, encoder.cfg.dim))
        self.motion_mlp = nn.Sequential(
            nn.Linear(encoder.cfg.dim, encoder.cfg.dim*2),
            nn.GELU(),
            nn.Linear(encoder.cfg.dim*2, motion_dim))
        self.motion_basis = nn.Parameter(torch.randn(encoder.cfg.dim, motion_dim) * 0.01)
        self.z_proj = nn.Linear(encoder.cfg.dim*2, decoder.cfg.dec_dim) 

    # --------------------- MAE helpers (same pathway as before) ---------------------

    @staticmethod
    def _patchify(x: torch.Tensor, patch: int) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        x: (B,T,C,H,W) -> (B,T,N,patch_dim), hw=(h,w)
        patch_dim = patch*patch*C
        """
        B, T, C, H, W = x.shape
        if H % patch != 0 or W % patch != 0:
            raise ValueError(f"H,W must be divisible by patch={patch}, got {(H,W)}")
        h, w = H // patch, W // patch
        x = x.view(B, T, C, h, patch, w, patch)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        x = x.view(B, T, h * w, patch * patch * C)
        return x, (h, w)

    @staticmethod
    def _random_keep_idx(B: int, T: int, N: int, Nvis: int, device) -> torch.Tensor:
        # Uniform random per-frame permutations, take first Nvis
        idx = torch.empty((B, T, Nvis), device=device, dtype=torch.long)
        for b in range(B):
            for t in range(T):
                idx[b, t] = torch.randperm(N, device=device)[:Nvis]
        return idx

    @staticmethod
    def _masked_idx_sorted(N: int, keep_bt: torch.Tensor) -> torch.Tensor:
        """
        keep_bt: (BT,Nvis)
        Returns mask_idx: (BT,Nmask) indices of masked patches in ascending order.
        """
        BT, Nvis = keep_bt.shape
        keep_bool = torch.zeros(BT, N, device=keep_bt.device, dtype=torch.bool)
        keep_bool.scatter_(1, keep_bt, True)
        mask_idx = (~keep_bool).nonzero(as_tuple=False)[:, 1]
        return mask_idx.view(BT, N - Nvis)

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
        return F.smooth_l1_loss(pred, target, reduction="mean")

    # ----------------------------------- forward -----------------------------------

    def forward(
        self,
        video: torch.Tensor,                      # (B,T,C,H,W)
        target: torch.Tensor | None = None,       # (B,T,C,H,W) optional target for frame loss (defaults to input video)
        *,
        keep_idx: torch.Tensor | None = None,     # (B,T,Nvis)
        mask_ratio: float | None = None,          # overrides self.mask_ratio if keep_idx is None
        return_pred: bool = False,
        return_frame_pred: bool = False,
    ):
        """
        return_pred:
          - returns MAE patch-assembled reconstruction (same as old wrapper)

        return_frame_pred:
          - returns CLS->frame reconstruction
        """
        B, T, C, H, W = video.shape

        if self.patch is None or self.in_chans is None:
            raise ValueError("Decoder must have attributes .patch and .in_chans (as in provided VideoViTDecoder).")
        if C != self.in_chans:
            raise ValueError(f"video C={C} must match decoder in_chans={self.in_chans}")

        # --- targets for MAE patch loss ---
        if target is None:
            target_full, hw = self._patchify(video, self.patch)  # (B,T,N,patch_dim)
            target_video = video
        else:
            target_full, hw = self._patchify(target, self.patch)  # (B,T,N,patch_dim)
            target_video = target
            
        h, w = hw
        N = target_full.size(2)
        patch_dim = target_full.size(3)

        # --- keep_idx (same logic as old wrapper) ---
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

        # --- encoder ---
        gcls, enc_tokens, hw_enc = self.encoder(video, keep_idx=keep_idx)
        if hw_enc != hw:
            raise ValueError(f"Encoder hw={hw_enc} != patchify hw={hw}")

        # enc_tokens: (B,T,1+Nvis,Denc)
        # per-frame CLS from encoder output
        frame_cls = enc_tokens[:, :, 0, :]  # (B,T,Denc)

        # --- (B) CLS -> low rank frame decoding ---
        # Global template
        z_template = self.template_mlp(gcls).unsqueeze(1).expand(-1, T, -1)  # (B,T,Denc)
        
        # Low Rank motion
        z_motion = self.motion_mlp(frame_cls)             # (B,T,motion_dim)
        Q, _ = torch.linalg.qr(self.motion_basis)
        delta_z = z_motion @ Q.T                        # (B,T,Denc)

        # Combine
        frame_z = z_template + delta_z                        # (B,T,Denc)

        pred_frames = self.frame_decoder(frame_z)  # (B,T,C*,Hf,Wf)

        if pred_frames.shape[2] != C:
            raise ValueError(
                f"frame_decoder output channels={pred_frames.shape[2]} must match input C={C} "
                f"(decoder in_chans={self.in_chans})."
            )
        if pred_frames.shape[-2:] != self.frame_target_size:
            raise ValueError(
                f"frame_decoder output spatial={tuple(pred_frames.shape[-2:])} "
                f"must equal frame_target_size={self.frame_target_size}."
            )
        loss_frame = self._loss(pred_frames, target_video)

        # --- (A) MAE masked-patch decoding (same as old wrapper) ---
        # Create "mask" token from global & frame information
        mask_tok = self.z_proj(torch.cat([gcls.unsqueeze(1).expand(-1, T, -1), frame_cls], dim=-1))
        mask_tok = mask_tok.view(B * T, -1).unsqueeze(1).expand(-1, N, -1)
        pred_masked = self.decoder(enc_tokens, mask_token=mask_tok, keep_idx=keep_idx, hw=hw)  # (B,T,Nmask,patch_dim)
        Nmask = N - Nvis

        # gather masked targets in the same order decoder uses
        BT = B * T
        target_bt = target_full.view(BT, N, patch_dim)
        keep_bt = keep_idx.view(BT, Nvis)
        mask_idx = self._masked_idx_sorted(N, keep_bt)                  # (BT,Nmask)
        target_masked_bt = self._gather_bt(target_bt, mask_idx)         # (BT,Nmask,D)
        target_masked = target_masked_bt.view(B, T, Nmask, patch_dim)

        if self.norm_pix_loss:
            mean = target_masked.mean(dim=-1, keepdim=True)
            var = target_masked.var(dim=-1, keepdim=True, unbiased=False)
            target_masked = (target_masked - mean) / (var + 1e-6).sqrt()

        loss_mae = self._loss(pred_masked, target_masked)

        # --- total ---
        loss = self.mae_loss_weight * loss_mae + self.frame_loss_weight * loss_frame

        out = {
            "loss": loss,
            "loss_mae": loss_mae,
            "loss_frame": loss_frame,
            "z_motion": z_motion,
        }

        # Optional: return patch-assembled reconstruction (old behavior)
        if return_pred:
            pred_masked_bt = pred_masked.view(BT, Nmask, patch_dim)

            pred_full_bt = target_bt.clone()  # visible patches copied from input for visualization
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

        # Optional: return CLS->frame reconstruction
        if return_frame_pred:
            out["pred_frames"] = pred_frames  # (B,T,C,frame_H,frame_W)

        return out


# ----------------------------- example -----------------------------
if __name__ == "__main__":
    enc = VideoViTEncoder(VideoViTCfg(dim=384, depth=8, heads=6, patch=8))
    dec = VideoViTDecoder(enc_dim=384, patch=8, in_chans=3, cfg=VideoViTDecCfg(dec_dim=256, dec_depth=2, dec_heads=8))
    mae = VideoMotionMAE(
        encoder=enc,
        decoder=dec,
        frame_decoder=SimpleConvDecoder(latent=384, out_dim=3, base=256),
        norm_pix_loss=True,
        loss_type="mse",
        mask_ratio=0.75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae = mae.to(device)
    video = torch.randn(16, 32, 3, 128, 128, device=device)  # (B,T,C,H,W)

    # Profile memory usage
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, 
                    torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        output = mae(video, return_pred=True)["pred"]

    assert output.shape == video.shape, f"Expected output shape {video.shape}, got {output.shape}"
    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in mae.parameters() if p.requires_grad)/1e6, 2), 'M')