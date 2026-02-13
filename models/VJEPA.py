import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .VideoViT import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg


class VideoJEPA(nn.Module):
    """
    JEPA-style masked prediction for video.

    Student:
      - encoder sees ONLY visible patches (via keep_idx)
      - decoder/predictor outputs predictions for MASKED patch positions

    Teacher (EMA of student encoder):
      - processes the FULL video with NO masking (keep all patches)
      - produces per-patch target encodings

    Loss:
      - L1(pred_masked, stopgrad(teacher_masked))

    Notes:
      - encoder.forward requires timestamps: (B, T)
      - decoder.forward requires timestamps: (B, T)
      - decoder is assumed to output *representations* (dim == encoder.cfg.dim)
        for the masked patches (NOT pixel-space patch vectors).
    """
    def __init__(
        self,
        encoder: VideoViTEncoder,
        decoder: VideoViTDecoder,
        momentum: float = 0.996,
        *,
        mask_ratio: float = 0.75,
        ema_init_from_student: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.momentum = momentum
        self.mask_ratio = float(mask_ratio)

        # ---- EMA teacher encoder (no grads) ----
        self.ema_encoder = copy.deepcopy(encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)
        self.ema_encoder.eval()

        if ema_init_from_student:
            self._copy_student_to_ema()

        # ---- patch/grid info ----
        # we need patch size to know N = (H/patch)*(W/patch) for masking / full keep_idx
        self.patch = getattr(decoder, "patch", None) or getattr(encoder, "patch", None)
        if self.patch is None:
            raise ValueError("Need .patch on encoder or decoder to compute patch grid size for masking.")

    # --------------------- EMA helpers ---------------------

    @torch.no_grad()
    def _copy_student_to_ema(self):
        """Hard copy student encoder weights/buffers into EMA encoder."""
        self.ema_encoder.load_state_dict(self.encoder.state_dict(), strict=True)

    @torch.no_grad()
    def update_ema(self):
        """
        EMA update: ema = m*ema + (1-m)*student

        Call this once per optimizer step (typically after student weights update).
        """
        ema_sd = self.ema_encoder.state_dict()
        stu_sd = self.encoder.state_dict()

        for k, ema_v in ema_sd.items():
            stu_v = stu_sd[k]
            # float tensors get EMA; non-floats (e.g., int buffers) are copied
            if torch.is_floating_point(ema_v):
                ema_v.mul_(self.momentum).add_(stu_v, alpha=(1.0 - self.momentum))
            else:
                ema_v.copy_(stu_v)

    # --------------------- masking helpers ---------------------

    @staticmethod
    def _random_keep_idx(B: int, T: int, N: int, Nvis: int, device) -> torch.Tensor:
        """
        Sample visible patches once per video (per b), then repeat across time.
        Output: (B,T,Nvis) with identical keep indices for all t.
        Vectorized + torch.compile friendly.
        """
        # scores: (B, N). argsort gives a permutation per row.
        scores = torch.rand((B, N), device=device)
        idx_b = torch.argsort(scores, dim=1)[:, :Nvis]  # (B, Nvis)
        return idx_b[:, None, :].expand(B, T, Nvis).contiguous()


    @staticmethod
    def _masked_idx_sorted(N: int, keep_bt: torch.Tensor) -> torch.Tensor:
        """
        keep_bt: (BT,Nvis)
        Returns mask_idx: (BT,Nmask) indices of masked patches in ascending order.
        """
        BT, Nvis = keep_bt.shape
        device = keep_bt.device
        keep_bool = torch.zeros((BT, N), device=device, dtype=torch.bool)
        keep_bool.scatter_(1, keep_bt, True)
        idx_all = torch.arange(N, device=device, dtype=torch.long).expand(BT, N)
        masked_candidates = torch.where(keep_bool, torch.full_like(idx_all, N), idx_all)
        masked_sorted, _ = torch.sort(masked_candidates, dim=1)
        Nmask = N - Nvis
        return masked_sorted[:, :Nmask]  # (BT, Nmask), already ascending


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


    # ----------------------------------- forward -----------------------------------

    def forward(
        self,
        video: torch.Tensor,                      # (B,T,C,H,W)
        timestamps: torch.Tensor,                 # (B,T)
        target: torch.Tensor | None = None,       # Optional teacher target
        keep_idx: torch.Tensor | None = None,     # (B,T,Nvis)
        mask_ratio: float | None = None,          # overrides self.mask_ratio if keep_idx is None
        return_debug: bool = False,
    ):
        B, T, C, H, W = video.shape

        if timestamps.dim() != 2 or timestamps.shape[:2] != (B, T):
            raise ValueError(f"timestamps must be (B,T)=({B},{T}), got {tuple(timestamps.shape)}")
        timestamps = timestamps.to(device=video.device)

        patch = int(self.patch)
        if H % patch != 0 or W % patch != 0:
            raise ValueError(f"H,W must be divisible by patch={patch}, got {(H, W)}")
        h, w = H // patch, W // patch
        N = h * w

        # --- keep_idx / masking ---
        if keep_idx is None:
            r = self.mask_ratio if mask_ratio is None else float(mask_ratio)
            if not (0.0 < r < 1.0):
                raise ValueError(f"mask_ratio must be in (0,1), got {r}")
            Nvis = max(1, int(round(N * (1.0 - r))))
            keep_idx = self._random_keep_idx(B, T, N, Nvis, video.device)
        else:
            if keep_idx.dim() != 3 or keep_idx.shape[:2] != (B, T):
                raise ValueError(f"keep_idx must be (B,T,Nvis)=({B},{T},*), got {tuple(keep_idx.shape)}")
            keep_idx = keep_idx.to(device=video.device, dtype=torch.long)
            Nvis = keep_idx.size(2)

        Nmask = N - Nvis
        BT = B * T

        # --- student encoder (masked input) ---
        gcls_s, enc_tokens_s, hw_s = self.encoder(video, keep_idx=keep_idx, timestamps=timestamps)
        if hw_s != (h, w):
            raise ValueError(f"Student encoder hw={hw_s} != expected hw={(h, w)} from patch/grid.")

        # --- teacher encoder (EMA, full / unmasked) ---
        full_keep = torch.arange(N, device=video.device, dtype=torch.long).view(1, 1, N)
        full_keep = full_keep.expand(B, T, N).contiguous()

        with torch.no_grad():
            target = video if target is None else target
            self.ema_encoder.eval()
            gcls_t, enc_tokens_t, hw_t = self.ema_encoder(target, keep_idx=full_keep, timestamps=timestamps)
            if hw_t != (h, w):
                raise ValueError(f"Teacher encoder hw={hw_t} != expected hw={(h, w)} from patch/grid.")
            # enc_tokens_t: (B,T,1+N,Denc)  -> patch tokens are [1:]
            teacher_patches = enc_tokens_t[:, :, 1:, :]  # (B,T,N,Denc)

        # --- student predictor/decoder: predict masked patch embeddings ---
        # Create per-position decoder mask token grid (B,T,N,Ddec) from frame_z
        pred_masked = self.decoder(
            gcls_s,
            enc_tokens_s,
            keep_idx=keep_idx,
            hw=(h, w),
            timestamps=timestamps,
        )  # expected: (B,T,Nmask,Denc) in JEPA mode

        if pred_masked.shape[:3] != (B, T, Nmask):
            raise ValueError(
                f"decoder must return (B,T,Nmask,Dim), got {tuple(pred_masked.shape)} with Nmask={Nmask}"
            )
        if pred_masked.size(-1) != self.encoder.cfg.dim:
            raise ValueError(
                f"JEPA expects decoder output dim == encoder.cfg.dim ({self.encoder.cfg.dim}), "
                f"got {pred_masked.size(-1)}. Configure VideoViTDecoder head to predict encoder-space embeddings."
            )

        # --- align masked indices (same ordering as in your jepa code) ---
        keep_bt = keep_idx.view(BT, Nvis)                  # (BT,Nvis)
        mask_idx = self._masked_idx_sorted(N, keep_bt)     # (BT,Nmask)

        teacher_bt = teacher_patches.reshape(BT, N, -1)    # (BT,N,Denc)
        teacher_masked_bt = self._gather_bt(teacher_bt, mask_idx)  # (BT,Nmask,Denc)
        teacher_masked = teacher_masked_bt.view(B, T, Nmask, -1).detach()  # stop-grad

        # --- JEPA loss (L1) ---
        loss = F.l1_loss(pred_masked, teacher_masked, reduction="mean")

        out = {
            "loss": loss,
        }

        if return_debug:
            out.update(
                {
                    "pred_masked": pred_masked,              # student predicted masked embeddings
                    "teacher_masked": teacher_masked,        # EMA teacher masked embeddings (detached)
                    "mask_idx": mask_idx.view(B, T, Nmask),  # masked positions
                }
            )

        return out


# ----------------------------- example -----------------------------
if __name__ == "__main__":
    import json
    config = json.load(open("config/VJEPA.json", "r"))
    enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
    dec = VideoViTDecoder(enc_dim=config["encoder"]["dim"], patch=config["encoder"]["patch"], 
                        in_chans=config["encoder"]["in_chans"], cfg=VideoViTDecCfg(**config["decoder"]))
    jepa = VideoJEPA(enc, dec, momentum=config["jepa"]["momentum"], mask_ratio=config["jepa"]["mask_ratio"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa = jepa.to(device)

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
            out = jepa(video, timestamps=timestamps)
            loss = out["loss"]
        loss.backward()

    print(prof.key_averages().table(sort_by=f"self_{device}_memory_usage", row_limit=8))
    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    print(
        "Total trainable parameters:",
        round(sum(p.numel() for p in jepa.parameters() if p.requires_grad) / 1e6, 2),
        "M",
    )
