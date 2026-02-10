import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from datahandling.PreTrainEchoDynaDataset import load_echonet_dynamic_datasets
from models.VideoViT import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg
from models.ViTMAE import VideoViTMAE
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%Y_%m_%d")
timestamp = datetime.now().strftime("%H_%M")
output_dir = f"results/{date}/{timestamp}_VMAE"
os.makedirs(output_dir, exist_ok=True)

# Training Parameters
epochs = 300
batch_size = 16
learning_rate = 2e-4
weight_decay = 1e-2
max_frames = 32

torch.set_float32_matmul_precision('high')
autocast = False

enc = VideoViTEncoder(VideoViTCfg(dim=384, depth=8, heads=6, patch=8))
dec = VideoViTDecoder(enc_dim=384, patch=8, in_chans=3, cfg=VideoViTDecCfg(dec_dim=256, dec_depth=2, dec_heads=8))
mae = VideoViTMAE(enc, dec, norm_pix_loss=False, mask_ratio=0.75)
mae = mae.to(device)
# mae = torch.compile(mae)
print(f"Initialized VMAE with {sum(p.numel() for p in mae.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")
optimizer = torch.optim.AdamW(mae.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Augmentations
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

class FrameDropout(nn.Module):
    """
    Randomly drops entire frames in the temporal dimension with probability p.
    Input shape: [B, T, C, H, W]
    """
    def __init__(self, p: float = 0.1):
        super(FrameDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, _, _ = x.shape
        mask = (torch.rand(B, T, device=x.device) > self.p).float()  # [B, T]
        mask = mask[:, :, None, None, None]  # [B, T, 1, 1, 1]
        return x * mask

class RandomVideoErasing(nn.Module):
    """
    Randomly zeros one or more spatiotemporal cuboids in a video tensor.

    Input:  x of shape [B, T, C, H, W]
    Output: same shape, with K [t, h, w] cuboids set to `value` (across all C).

    Similar to torchvision.transforms.RandomErasing, but extended to video and
    supports multiple cuboids.
    """

    def __init__(
        self,
        p: float = 0.5,
        scale= (0.02, 0.4),
        ratio = (0.3, 3.3),
        t_scale = (0.1, 1.0),
        value = 0.0,
        per_sample: bool = True,
        attempts: int = 10,
        inplace: bool = False,
        num_cuboids: int = 1,
        num_cuboids_range = None,
    ):
        """
        Args:
            p: probability of applying erasing.
            scale: fraction of spatial area to erase (relative to H*W) per cuboid.
            ratio: aspect ratio range (w/h) per cuboid.
            t_scale: fraction of temporal length to erase (relative to T) per cuboid.
            value: fill value (0.0 requested).
            per_sample: if True, each batch element gets its own cuboids.
                        if False, same cuboids are applied to all batch elements.
            attempts: number of tries to find a valid region for each cuboid.
            inplace: if True, modify input in place.
            num_cuboids: fixed number of cuboids to erase (ignored if num_cuboids_range is set).
            num_cuboids_range: if set, sample K uniformly from [low, high] (inclusive).
        """
        super().__init__()

        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        if scale[0] <= 0 or scale[1] <= 0 or scale[0] > scale[1]:
            raise ValueError("scale must be (min, max) with 0 < min <= max.")
        if ratio[0] <= 0 or ratio[0] > ratio[1]:
            raise ValueError("ratio must be (min, max) with 0 < min <= max.")
        if t_scale[0] <= 0 or t_scale[1] <= 0 or t_scale[0] > t_scale[1]:
            raise ValueError("t_scale must be (min, max) with 0 < min <= max.")
        if attempts < 1:
            raise ValueError("attempts must be >= 1.")
        if num_cuboids_range is not None:
            lo, hi = num_cuboids_range
            if lo < 0 or hi < lo:
                raise ValueError("num_cuboids_range must be (low, high) with 0 <= low <= high.")
        else:
            if num_cuboids < 0:
                raise ValueError("num_cuboids must be >= 0.")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.t_scale = t_scale
        self.value = float(value)
        self.per_sample = per_sample
        self.attempts = attempts
        self.inplace = inplace
        self.num_cuboids = num_cuboids
        self.num_cuboids_range = num_cuboids_range

    @staticmethod
    def _rand_uniform(a: float, b: float) -> float:
        return a + (b - a) * random.random()

    def _sample_k(self) -> int:
        if self.num_cuboids_range is None:
            return self.num_cuboids
        lo, hi = self.num_cuboids_range
        return random.randint(lo, hi)

    def _sample_region(self, T: int, H: int, W: int):
        """
        Returns (t0, t1, y0, y1, x0, x1) or None if couldn't find a valid region.
        """
        area = H * W

        for _ in range(self.attempts):
            # temporal length
            t_frac = self._rand_uniform(self.t_scale[0], self.t_scale[1])
            t_len = max(1, int(round(t_frac * T)))
            if t_len > T:
                continue

            # spatial rectangle (area + aspect ratio)
            erase_area = self._rand_uniform(self.scale[0], self.scale[1]) * area
            aspect = self._rand_uniform(self.ratio[0], self.ratio[1])

            h = int(round((erase_area / aspect)**0.5))
            w = int(round((erase_area * aspect)**0.5))

            if h < 1 or w < 1 or h > H or w > W:
                continue

            t0 = random.randint(0, T - t_len)
            y0 = random.randint(0, H - h)
            x0 = random.randint(0, W - w)

            return (t0, t0 + t_len, y0, y0 + h, x0, x0 + w)

        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")

        if (not self.training) or (random.random() > self.p):
            return x

        B, T, C, H, W = x.shape
        out = x if self.inplace else x.clone()

        def erase_on_tensor(tensor_view: torch.Tensor) -> None:
            # tensor_view is [T, C, H, W] for one sample, or [B, T, C, H, W] for whole batch
            K = self._sample_k()
            for _ in range(K):
                region = self._sample_region(T, H, W)
                if region is None:
                    continue
                t0, t1, y0, y1, x0, x1 = region

                if tensor_view.ndim == 4:
                    # [T, C, H, W]
                    tensor_view[t0:t1, :, y0:y1, x0:x1] = self.value
                else:
                    # [B, T, C, H, W]
                    tensor_view[:, t0:t1, :, y0:y1, x0:x1] = self.value

        if self.per_sample:
            for b in range(B):
                erase_on_tensor(out[b])
        else:
            erase_on_tensor(out)

        return out

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
])


# Functions
def plot_recons(mae, val_ds, output_dir, device, clip_len=16, stride=2):
    mae.eval()

    # --- sample one random validation video ---
    vid_idx = int(torch.randint(len(val_ds), (1,)).item())
    video = val_ds[vid_idx]["video"]  # [C, T, H, W]
    T = video.shape[1]

    # --- choose start so we can grab clip_len frames (or as many as available) ---
    if T >= clip_len:
        start = int(torch.randint(0, T - clip_len + 1, (1,)).item())
        clip = video[:, start : start + clip_len]  # [C, clip_len, H, W]
    else:
        clip = video  # [C, T, H, W]
        clip_len = T  # adjust
    clip = clip.transpose(0, 1)  # -> [T, C, H, W]

    # --- reconstruct full clip once, then subsample both orig and recon ---
    with torch.no_grad():
        clip_batch = clip.unsqueeze(0).to(device)
        pred = mae(clip_batch, return_pred=True)["pred"]
        recon = pred.squeeze(0).cpu()  # -> [C, T, H, W]

    clip = clip.cpu()

    # drop every other frame (or whatever stride is)
    clip = clip[::stride]
    recon = recon[::stride]
    n_cols = clip.shape[0]

    def to_numpy(img_t):
        img_t = (img_t + 1.0) / 2.0
        arr = img_t.permute(1, 2, 0).numpy()
        arr = arr.clip(0.0, 1.0)
        return arr[:, :, 0] if arr.shape[2] == 1 else arr

    # --- plot ---
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 2, 4))
    for i in range(n_cols):
        orig_np = to_numpy(clip[i])
        recon_np = to_numpy(recon[i])

        axs[0, i].axis("off")
        axs[1, i].axis("off")

        if orig_np.ndim == 2:
            axs[0, i].imshow(orig_np, cmap="gray")
            axs[1, i].imshow(recon_np, cmap="gray")
        else:
            axs[0, i].imshow(orig_np)
            axs[1, i].imshow(recon_np)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/recon_mae.png", bbox_inches="tight")
    plt.close(fig)

@torch.no_grad()
def median_blur(x: torch.Tensor, padding_mode: str = "reflect") -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected x with 5 dims [B,T,C,H,W], got {tuple(x.shape)}")

    B, T, C, H, W = x.shape
    if H < 1 or W < 1:
        return x

    # Treat each time-step as an independent image in the batch
    # [B,T,C,H,W] -> [B*T, C, H, W]
    xt = x.contiguous().view(B * T, C, H, W)

    # Pad H,W by 1 on each side
    xt = F.pad(xt, pad=(1, 1, 1, 1), mode=padding_mode)

    # Unfold 3x3 neighborhoods: output is [B*T, C*9, H*W]
    patches = F.unfold(xt, kernel_size=3, dilation=1, padding=0, stride=1)

    # Reshape to [B*T, C, 9, H*W] so we can take per-channel median
    patches = patches.view(B * T, C, 9, H * W)

    # Median of 9 values = 5th smallest (k=5, 1-indexed)
    med = patches.kthvalue(k=5, dim=2).values  # [B*T, C, H*W]

    # Back to [B, T, C, H, W]
    out = med.view(B, T, C, H, W).contiguous()
    return out


# Dataset
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets()

def collate_fn(batch):
    # Random sample to the minimum number of frames in the batch, with max cap
    min_frames = min(min(item['video'].shape[1] for item in batch), max_frames)
    for item in batch:
        T = item['video'].shape[1]
        if T > min_frames:
            max_start = T - min_frames
            start = torch.randint(0, max_start + 1, (1,)).item()
            item['video'] = item['video'][:, start:start + min_frames, :, :]
        else:
            item['video'] = item['video'][:, :min_frames, :, :]
    videos = torch.stack([item['video'] for item in batch])  # [B, C, T, H, W]
    videos = videos.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
    return {'video': videos}

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                      collate_fn=collate_fn,
                      num_workers=60, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True,
                    collate_fn=collate_fn)


# Training 
train_losses = []; val_losses = []
for epoch in range(epochs):
    mae.train()
    train_loss = 0.0; pred_loss = 0.0
    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in p_bar:
        videos = batch['video'].to(device, non_blocking=True)  # [B, C, T, H, W]

        # Augmentations
        aug_videos = augmentations(videos)
        videos = median_blur(videos)  # optional denoise targets
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
            loss = mae(aug_videos, target=videos, return_pred=False)["loss"]

        loss.backward()
        norm = nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
        optimizer.step()
    
        train_loss += loss.item() * videos.size(0)
        p_bar.set_postfix({'Recon': loss.item(), 'Grad Norm': norm.item()})
        
    train_loss /= len(train_dl.dataset)
    train_losses.append(train_loss)
    
    mae.eval()
    val_loss = 0.0
    with torch.no_grad():
        p_bar = tqdm(val_dl, desc=f"Validation Epoch {epoch+1}/{epochs}")
        for batch in p_bar:
            videos = batch['video'].to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
                loss = mae(videos, return_pred=False)["loss"]
            val_loss += loss.item() * videos.size(0)
            p_bar.set_postfix({'Recon': loss.item()})
            
    val_loss /= len(val_dl.dataset)
    val_losses.append(val_loss)    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    # Saving mae, reconstructions, losses
    torch.save(mae.state_dict(), f"{output_dir}/VMAE.pth")
    plot_recons(mae, val_ds, output_dir, device)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, epoch + 2), train_losses, label='Training', color='tab:blue')
    ax.plot(range(1, epoch + 2), val_losses, label='Validation', color='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Losses over Epochs')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/losses.png", bbox_inches="tight")
    plt.close()
