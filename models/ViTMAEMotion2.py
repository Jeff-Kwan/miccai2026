import torch
import torch.nn as nn
import torch.nn.functional as F

from .VideoViT2 import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg


class SimpleConvDecoder(nn.Module):
    """
    Input:  z of shape [B, T, latent]  (CLS tokens per frame)
    Output: x of shape [B, T, out_dim, H, W]
    """
    def __init__(self, latent: int, out_dim: int = 3, base: int = 256):
        super().__init__()

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),   # 1->2->4->...->128
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.GELU(),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, base, 2, 2, 0), # 1 -> 2
            up_block(base,      base // 2),   # 2   -> 4
            up_block(base // 2, base // 4),   # 4   -> 8
            up_block(base // 4, base // 8),   # 8   -> 16
            up_block(base // 8, base // 16),  # 16  -> 32
            up_block(base // 16, base // 32), # 32  -> 64
            nn.Conv2d(base // 32, out_dim, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, H, W) -> torch.Tensor:
        # z: (B, T, latent)
        B, T, latent = z.shape
        z = z.view(B * T, latent, 1, 1)     # (B*T, latent, 1, 1)
        x = self.decoder(z)                 # (B*T, out_dim, 64, 64)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) 
        return x.view(B, T, -1, H, W)  # (B, T, out_dim, H, W)


class VideoMotionMAE(nn.Module):
    """
    Video MAE with two reconstruction pathways:

      (A) MAE patch pathway:
          - keep_idx selects visible patches (constant Nvis across frames)
          - encoder encodes only visible patches
          - decoder reconstructs masked patches only
          - loss computed on masked patches only

      (B) CLS -> frame pathway:
          - take per-frame CLS tokens from encoder output
          - decode them into full frames using a frame_decoder
          - compute a frame reconstruction loss against the input frames

    NOTE (compat w/ new temporal RoPE):
      - encoder.forward now requires timestamps: (B,T)
      - decoder.forward now requires timestamps: (B,T)
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

        if loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(f"loss_type must be mse|l1|smooth_l1, got {loss_type}")

        # Convenience: pull patch/in_chans from decoder if present
        self.patch = getattr(decoder, "patch", None)
        self.in_chans = getattr(decoder, "in_chans", None)

        # Motion!
        self.template_mlp = nn.Sequential(
            nn.Linear(encoder.cfg.dim, encoder.cfg.dim * 2),
            nn.GELU(),
            nn.Linear(encoder.cfg.dim * 2, encoder.cfg.dim),
        )
        self.motion_mlp = nn.Sequential(
            nn.Linear(encoder.cfg.dim, encoder.cfg.dim * 2),
            nn.GELU(),
            nn.Linear(encoder.cfg.dim * 2, motion_dim),
        )
        self.motion_basis = nn.Parameter(torch.randn(encoder.cfg.dim, motion_dim) * 0.01)

        # used to create decoder-dim tokens for all N positions
        self.z_proj = nn.Linear(encoder.cfg.dim * 2, decoder.cfg.dec_dim)

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

    def low_rank_latent(self, gcls, frame_cls, T):
        z_template = self.template_mlp(gcls).unsqueeze(1).expand(-1, T, -1)  # (B,T,Denc)
        z_motion = self.motion_mlp(frame_cls)  # (B,T,motion_dim)
        Q, _ = torch.linalg.qr(self.motion_basis)  # (Denc, motion_dim) -> orthonormal columns
        delta_z = z_motion @ Q.T  # (B,T,Denc)
        frame_z = z_template + delta_z  # (B,T,Denc)
        return frame_z, z_motion

    def forward(
        self,
        video: torch.Tensor,                      # (B,T,C,H,W)
        timestamps: torch.Tensor,                 # (B,T) REQUIRED for temporal RoPE
        target: torch.Tensor | None = None,       # (B,T,C,H,W) optional target for frame loss
        *,
        keep_idx: torch.Tensor | None = None,     # (B,T,Nvis)
        mask_ratio: float | None = None,          # overrides self.mask_ratio if keep_idx is None
        return_pred: bool = False,
    ):
        """
        return_pred:
          - returns MAE patch-assembled reconstruction

        return_frame_pred:
          - returns CLS->frame reconstruction
        """
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
            target_full, hw = self._patchify(video, self.patch)  # (B,T,N,patch_dim)
            target_video = video
        else:
            target_full, hw = self._patchify(target, self.patch)  # (B,T,N,patch_dim)
            target_video = target

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

        # --- encoder (NOW requires timestamps) ---
        gcls, enc_tokens, hw_enc = self.encoder(video, keep_idx=keep_idx, timestamps=timestamps)
        if hw_enc != hw:
            raise ValueError(f"Encoder hw={hw_enc} != patchify hw={hw}")

        # enc_tokens: (B,T,1+Nvis,Denc)
        frame_cls = enc_tokens[:, :, 0, :]  # (B,T,Denc)

        # --- (B) CLS -> frame decoding ---
        frame_z, z_motion = self.low_rank_latent(gcls, frame_cls, T)  # (B,T,Denc)
        pred_frames = self.frame_decoder(frame_z, H, W)  # (B,T,C,Hf,Wf)

        if pred_frames.shape[2] != C:
            raise ValueError(
                f"frame_decoder output channels={pred_frames.shape[2]} must match input C={C} "
                f"(decoder in_chans={self.in_chans})."
            )
        loss_frame = self._loss(pred_frames, target_video)

        # --- (A) MAE masked-patch decoding ---
        # Create a full (B,T,N,Ddec) "mask token grid" from global & frame information.
        # Decoder expects mask_token shaped (B, T, N, Ddec).
        g_rep = gcls.unsqueeze(1).expand(-1, T, -1)                 # (B,T,Denc)
        pair = torch.cat([g_rep, frame_cls], dim=-1)                # (B,T,2*Denc)
        base_tok = self.z_proj(pair)                                # (B,T,Ddec)
        mask_tok = base_tok.unsqueeze(2).expand(-1, -1, N, -1)      # (B,T,N,Ddec)
        mask_tok = mask_tok.reshape(B * T, N, -1)

        pred_masked = self.decoder(
            enc_tokens,
            keep_idx=keep_idx,
            hw=hw,
            mask_token=mask_tok,
            timestamps=timestamps,
        )  # (B,T,Nmask,patch_dim)

        Nmask = N - Nvis
        BT = B * T
        target_bt = target_full.view(BT, N, patch_dim)
        keep_bt = keep_idx.view(BT, Nvis)

        # gather masked targets in the same order decoder uses
        mask_idx = self._masked_idx_sorted(N, keep_bt)             # (BT,Nmask)
        target_masked_bt = self._gather_bt(target_bt, mask_idx)    # (BT,Nmask,patch_dim)
        target_masked = target_masked_bt.view(B, T, Nmask, patch_dim)

        if self.norm_pix_loss:
            mean = target_masked.mean(dim=-1, keepdim=True)
            var = target_masked.var(dim=-1, keepdim=True, unbiased=False)
            target_masked = (target_masked - mean) / (var + 1e-6).sqrt()

        loss_mae = self._loss(pred_masked, target_masked)

        loss = self.mae_loss_weight * loss_mae + self.frame_loss_weight * loss_frame

        out = {
            "loss": loss,
            "loss_mae": loss_mae,
            "loss_frame": loss_frame,
            "z_motion": z_motion,
        }

        # Optional: return patch-assembled reconstruction
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
            out["pred_frames"] = pred_frames  # (B,T,C,frame_H,frame_W)

        return out


# ----------------------------- example -----------------------------
if __name__ == "__main__":
    import json
    config = json.load(open("config/VMAE.json", "r"))
    enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
    dec = VideoViTDecoder(enc_dim=config["encoder"]["dim"], patch=config["encoder"]["patch"], 
                        in_chans=config["encoder"]["in_chans"], cfg=VideoViTDecCfg(**config["decoder"]))
    frame_dec = SimpleConvDecoder(latent=config["encoder"]["dim"], out_dim=config["encoder"]["in_chans"], base=config["decoder"]["dec_dim"])
    mae = VideoMotionMAE(enc, dec, frame_dec, motion_dim=2, norm_pix_loss=False, mask_ratio=0.75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae = mae.to(device)

    B, T, C, H, W = 32, 64, 3, 112, 112
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
