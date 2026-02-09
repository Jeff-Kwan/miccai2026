import torch
import torch.nn as nn
import torch.nn.functional as F
from .VideoViT import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg


class SimpleConvDecoder(nn.Module):
    def __init__(self, latent: int, out_dim: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent, 256, (1, 2, 2), (1, 2, 2)),  # 2x2
            nn.GELU(),
            nn.ConvTranspose3d(256, 256, (1, 2, 2), (1, 2, 2)),     # 4x4
            nn.GELU(),
            nn.ConvTranspose3d(256, 128, (1, 2, 2), (1, 2, 2)),     # 8x8
            nn.Conv3d(128, 128, (1, 3, 3), 1, (0, 1, 1)),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, (1, 2, 2), (1, 2, 2)),      # 16x16
            nn.GELU(),
            nn.ConvTranspose3d(64, 32, (1, 2, 2), (1, 2, 2)),       # 32x32
            nn.GELU(),
            nn.ConvTranspose3d(32, 8, (1, 2, 2), (1, 2, 2)),        # 64x64
            nn.ConvTranspose3d(8, out_dim, (1, 2, 2), (1, 2, 2)),   # 128x128
        )
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x is [B, T, latent]
            x = x.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).contiguous()  # (B, latent, T, 1, 1)
            return self.decoder(x).transpose(1, 2).contiguous()  # (B, out_dim, T, H, W)

class VideoViTMAE(nn.Module):
    """
    Video MAE wrapper around VideoViTEncoder + VideoViTDecoder.

    Training flow:
      1) Sample keep_idx (visible patch indices) per (B,T) with constant Nvis
      2) Encoder encodes only visible patches (no padding, no SDPA masks)
      3) Decoder reconstructs masked patches only
      4) Loss compares predicted masked patches vs ground-truth masked patches

    Notes:
      - keep_idx is the ONLY masking signal
      - Nvis is assumed constant across frames
      - Decoder output is masked patches in ascending patch-index order (0..N-1 excluding keep_idx),
        matching how targets are gathered below.
    """

    def __init__(
        self,
        encoder: nn.Module,                 # VideoViTEncoder
        decoder: nn.Module,                 # VideoViTDecoder
        *,
        norm_pix_loss: bool = False,
        loss_type: str = "mse",             # "mse" | "l1" | "smooth_l1"
        mask_ratio: float = 0.75,           # used only if keep_idx not provided
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.norm_pix_loss = norm_pix_loss
        self.loss_type = loss_type
        self.mask_ratio = mask_ratio

        if loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(f"loss_type must be mse|l1|smooth_l1, got {loss_type}")

        # Convenience: pull patch/in_chans from decoder if present
        self.patch = getattr(decoder, "patch", None)
        self.in_chans = getattr(decoder, "in_chans", None)

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
        # (B,T,C,h,ph,w,pw) -> (B,T,h,w,ph,pw,C) -> (B,T,N,ph*pw*C)
        x = x.view(B, T, C, h, patch, w, patch)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        x = x.view(B, T, h * w, patch * patch * C)
        return x, (h, w)

    @staticmethod
    def _random_keep_idx(B: int, T: int, N: int, Nvis: int, device) -> torch.Tensor:
        # Uniform random per-frame permutations, take first Nvis
        # (B,T,Nvis)
        idx = torch.empty((B, T, Nvis), device=device, dtype=torch.long)
        for b in range(B):
            for t in range(T):
                idx[b, t] = torch.randperm(N, device=device)[:Nvis]
        return idx

    @staticmethod
    def _masked_idx_sorted(N: int, keep_bt: torch.Tensor) -> torch.Tensor:
        """
        keep_bt: (BT,Nvis) indices in [0,N-1]
        Returns mask_idx: (BT,Nmask) indices of masked patches in ascending order.
        """
        BT, Nvis = keep_bt.shape
        keep_bool = torch.zeros(BT, N, device=keep_bt.device, dtype=torch.bool)
        keep_bool.scatter_(1, keep_bt, True)
        # nonzero gives ascending by column index when keep_bool built per-row
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

    def forward(
        self,
        video: torch.Tensor,                      # (B,T,C,H,W)
        *,
        keep_idx: torch.Tensor | None = None,     # (B,T,Nvis)
        mask_ratio: float | None = None,          # overrides self.mask_ratio if keep_idx is None
        return_pred: bool = False,
    ):
        B, T, C, H, W = video.shape

        if self.patch is None or self.in_chans is None:
            raise ValueError("Decoder must have attributes .patch and .in_chans (as in provided VideoViTDecoder).")
        if C != self.in_chans:
            raise ValueError(f"video C={C} must match decoder in_chans={self.in_chans}")

        # Targets (full patch grid)
        target_full, hw = self._patchify(video, self.patch)  # (B,T,N,patch_dim)
        h, w = hw
        N = target_full.size(2)
        D = target_full.size(3)

        # keep_idx
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

        # Encoder: returns (gcls, enc_tokens, hw_from_encoder)
        gcls, enc_tokens, hw_enc = self.encoder(video, keep_idx=keep_idx)
        if hw_enc != hw:
            raise ValueError(f"Encoder hw={hw_enc} != patchify hw={hw}")

        # Decoder: predicts masked patches only (B,T,Nmask,patch_dim)
        pred_masked = self.decoder(enc_tokens, keep_idx=keep_idx, hw=hw)  # (B,T,Nmask,D)
        Nmask = N - Nvis

        # Assemble full patch predictions: visible = input patches, masked = decoder preds
        # Work in BT for easy scatter
        BT = B * T
        target_bt = target_full.view(BT, N, D)           # (BT,N,D)
        keep_bt = keep_idx.view(BT, Nvis)                # (BT,Nvis)

        mask_idx = self._masked_idx_sorted(N, keep_bt)   # (BT,Nmask), ascending masked indices
        pred_masked_bt = pred_masked.view(BT, Nmask, D)  # (BT,Nmask,D)

        pred_full_bt = target_bt.clone()                 # start with visible patches from input
        pred_full_bt.scatter_(
            1,
            mask_idx.unsqueeze(-1).expand(BT, Nmask, D),
            pred_masked_bt,
        )
        pred_full_patches = pred_full_bt.view(B, T, N, D)  # (B,T,N,D)

        # Compute loss against masked targets (same as before)
        target_masked_bt = self._gather_bt(target_bt, mask_idx)          # (BT,Nmask,D)
        target_masked = target_masked_bt.view(B, T, Nmask, D)

        if self.norm_pix_loss:
            mean = target_masked.mean(dim=-1, keepdim=True)
            var = target_masked.var(dim=-1, keepdim=True, unbiased=False)
            target_masked = (target_masked - mean) / (var + 1e-6).sqrt()

        loss = self._loss(pred_masked, target_masked)

        if not return_pred:
            return {"loss": loss}

        # Unpatchify to video layout (B,T,C,H,W)
        pred_video = pred_full_patches.view(B, T, h, w, self.patch, self.patch, C)
        pred_video = pred_video.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        pred_video = pred_video.view(B, T, C, h * self.patch, w * self.patch)

        return {
            "loss": loss,
            "pred": pred_video,   # (B,T,C,H,W) full assembled reconstruction
        }



# ----------------------------- example -----------------------------
if __name__ == "__main__":
    enc = VideoViTEncoder(VideoViTCfg(dim=384, depth=8, heads=6, patch=8))
    dec = VideoViTDecoder(enc_dim=384, patch=8, in_chans=3, cfg=VideoViTDecCfg(dec_dim=256, dec_depth=2, dec_heads=8))
    mae = VideoViTMAE(enc, dec, norm_pix_loss=True, mask_ratio=0.75)

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
        loss = mae(video, return_pred=False)

    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in mae.parameters() if p.requires_grad)/1e6, 2), 'M')