import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from datahandling.PreTrainEchoDynaDataset import load_echonet_dynamic_datasets
from models.MotionLatentAE2 import MotionLatentAE
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%Y_%m_%d")
timestamp = datetime.now().strftime("%H_%M")
output_dir = f"results/{date}/{timestamp}_MLAE"
os.makedirs(output_dir, exist_ok=True)

# Training Parameters
epochs = 100
batch_size = 32
learning_rate = 3e-4
weight_decay = 1e-2
max_frames = 32
LAMBDAlat = 2e-3    # M rank
LAMBDAz = 1.0       # Z deviation
slow_factor = 0.02  # for motion_basis updates

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
autocast = False

model = MotionLatentAE(in_c=3, out_c=3, latent=256, enc_layers=4, 
                           dec_layers=2, levels=5, skips=False)
model = model.to(device)
# model = torch.compile(model)
print(f"Initialized MLAE with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")


# Augmentations
class FPSJitter(nn.Module):
    def __init__(self, k, min_keep=1, p=0.5):
        super().__init__()
        self.p = p
        self.k = k
        self.min_keep = min_keep

    def forward(self, x):
        if torch.rand(1).item() > self.p:
            return x
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        if self.min_keep >= T:
            return x
        k = torch.rand(1, device=x.device) * (self.k[1] - self.k[0]) + self.k[0]
        keep = (torch.rand(T, device=x.device) > k)  # [T] bool
        keep = keep.to(torch.bool)
        # always keep the first frame
        keep[0] = True
        if keep.sum().item() < self.min_keep:
            required = int(self.min_keep - keep.sum().item())
            if T - 1 > 0:
                idx = torch.randperm(T - 1, device=x.device)[:required] + 1  # choose from 1..T-1
                keep[idx] = True
        return x[:, :, keep, :, :]

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

augmentations = v2.Identity()
# augmentations = v2.Compose([
#     v2.RandomApply([# Intensities
#         v2.RandomChoice([
#             v2.RandomChoice([# Intensity distribution
#                 ClipBrightnessContrast(brightness=0.3, contrast=0.2),
#                 RandomGamma(gamma=(0.7, 1.5))]),
#             v2.RandomChoice([# Sharpness / Blur
#                 v2.RandomAdjustSharpness(sharpness_factor=0.5, p=1),
#                 v2.GaussianBlur(kernel_size=7, sigma=(0.25, 1.5))]),
#             v2.RandomChoice([# Noise
#                 v2.GaussianNoise(0, 0.05),
#                 SpeckleNoise(std=(0.02, 0.1))])
#         ])
#     ], p=0.5),
#     v2.RandomApply([# Masking
#         v2.RandomChoice([
#             FrameDropout(p=0.25),
#             RandomVideoErasing(p=1.0,
#             scale=(0.1, 0.25), ratio=(0.25, 4.0), t_scale=(1.0, 1.0), 
#             value=0.0, num_cuboids_range=(1, 3))])
#     ], p=0.0)
# ])

# fps_jitter = FPSJitter(k=(0.1, 0.75), min_keep=8, p=0.2)
fps_jitter = v2.Identity()

# Functions
def plot_recons(model, val_ds, output_dir):
    n_cols = 8
    # sample 8 random items from the validation dataset
    idxs = torch.randperm(len(val_ds))[:n_cols]
    videos_list = [val_ds[int(i)]['video'] for i in idxs]  # each: [C, T, H, W]

    # pick a random frame from each and make single-frame videos [C,1,H,W]
    samples = []
    for v in videos_list:
        T = v.shape[1]
        f = random.randint(0, T - 1)
        samples.append(v[:, f : f + 1, :, :])
    batch = torch.stack(samples)  # [B, C, 1, H, W]

    # run through model
    model.eval()
    with torch.no_grad():
        batch_device = batch.to(device)
        recon_batch = model(batch_device)
        recon_batch = recon_batch.cpu()
    batch = batch.cpu()

    # helper to convert tensor [C,H,W] -> numpy HxW or HxWx3
    def to_numpy(img_t):
        # [-1, 1] to [0, 1]
        img_t = (img_t + 1.0) / 2.0
        arr = img_t.permute(1, 2, 0).numpy()
        arr = arr.clip(0.0, 1.0)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr

    # plot 2 rows x 8 cols: top original, bottom reconstructed
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 2, 4))
    for i in range(n_cols):
        orig = batch[i][:, 0, :, :]       # [C, H, W]
        recon = recon_batch[i][:, 0, :, :]  # [C, H, W]
        orig_np = to_numpy(orig)
        recon_np = to_numpy(recon)

        axs[0, i].axis("off")
        axs[1, i].axis("off")
        if orig_np.ndim == 2:
            axs[0, i].imshow(orig_np, cmap="gray")
            axs[1, i].imshow(recon_np, cmap="gray")
        else:
            axs[0, i].imshow(orig_np)
            axs[1, i].imshow(recon_np)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/recon_frames.png", bbox_inches="tight")
    plt.clf()

    # Then pick a video to recon 16 consecutive frames
    vid_idx = idxs[0]
    video = val_ds[int(vid_idx)]['video']  # [C, T, H, W]
    T = video.shape[1]
    if T >= 16:
        start_frame = random.randint(0, T - 16)
    else:
        start_frame = 0
    clip = video[:, start_frame : start_frame + 16, :, :]  # [C, 16, H, W]
    clip_batch = clip.unsqueeze(0).to(device)  # [1, C, 16, H, W]
    model.eval()
    with torch.no_grad():
        recon_clip_batch = model(clip_batch)
        recon_clip_batch = recon_clip_batch.cpu()  # [1, C, 16, H, W]
    clip = clip.cpu()
    recon_clip = recon_clip_batch[0]  # [C, 16, H, W]
    fig, axs = plt.subplots(2, 16, figsize=(32, 4))
    for i in range(16):
        orig = clip[:, i, :, :]       # [C, H, W]
        recon = recon_clip[:, i, :, :]  # [C, H, W]
        orig_np = to_numpy(orig)
        recon_np = to_numpy(recon)

        axs[0, i].axis("off")
        axs[1, i].axis("off")
        if orig_np.ndim == 2:
            axs[0, i].imshow(orig_np, cmap="gray")
            axs[1, i].imshow(recon_np, cmap="gray")
        else:
            axs[0, i].imshow(orig_np)
            axs[1, i].imshow(recon_np)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recon_video.png", bbox_inches="tight")    
    plt.close()

@torch.no_grad()  # remove if you need gradients (median is piecewise-constant anyway)
def median_blur(x: torch.Tensor, padding_mode: str = "reflect") -> torch.Tensor:
    """
    3x3 median blur over H,W for x shaped [B, C, T, H, W].
    Median is computed independently per (B,C,T) plane.

    padding_mode: "reflect" (default), "replicate", or "circular".
    """
    if x.ndim != 5:
        raise ValueError(f"Expected x with 5 dims [B,C,T,H,W], got {tuple(x.shape)}")
    B, C, T, H, W = x.shape
    if H < 1 or W < 1:
        return x

    # Treat each time-step as an independent image in the batch
    # [B,C,T,H,W] -> [B*T, C, H, W]
    xt = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

    # Pad H,W by 1 on each side
    xt = F.pad(xt, pad=(1, 1, 1, 1), mode=padding_mode)

    # Unfold 3x3 neighborhoods: output is [B*T, C*9, H*W]
    patches = F.unfold(xt, kernel_size=3, dilation=1, padding=0, stride=1)

    # Reshape to [B*T, C, 9, H*W] so we can take per-channel median
    patches = patches.view(B * T, C, 9, H * W)

    # Median of 9 values = 5th smallest (k=5, 1-indexed)
    med = patches.kthvalue(k=5, dim=2).values  # [B*T, C, H*W]

    # Back to [B, C, T, H, W]
    out = med.view(B * T, C, H, W).view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
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
    return {'video': videos}

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                      collate_fn=collate_fn,
                      num_workers=80, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=32,
                    collate_fn=collate_fn)


# Training 
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW([
    {'params': model.motion_basis, 'lr': learning_rate * slow_factor, 'weight_decay': weight_decay * slow_factor},
    {'params': [p for n, p in model.named_parameters() if 'motion_basis' not in n], 'lr': learning_rate, 'weight_decay': weight_decay * 0.1}
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

train_losses = []; val_losses = []; e_ranks = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in p_bar:
        videos = batch['video'].to(device, non_blocking=True)  # [B, C, T, H, W]

        # Augmentations
        videos = fps_jitter(videos)
        aug_videos = augmentations(videos.transpose(1, 2)).transpose(1, 2).contiguous()

        optimizer.zero_grad()

        # Pad to max_frames if needed
        # T = aug_videos.size(2)
        # if T < max_frames:
        #     pad = (0, 0, 0, 0, 0, max_frames - T)  # pad T dimension at the end
        #     aug_videos = F.pad(aug_videos, pad, mode='constant', value=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
            x_rec = model(aug_videos)  # [B, C, T, H, W]
            mse_loss = criterion(x_rec, videos)
            loss = mse_loss + LAMBDAz * model.z_reg + LAMBDAlat * model.latent_reg

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
        train_loss += mse_loss.item() * videos.size(0)
        p_bar.set_postfix({'Recon': mse_loss.item(), 
                           'Zrank': model.effective_rank.item(), 
                           'Mrank': model.latent_reg.exp().item(),
                           'ZDev': model.z_reg.item(),
                           'Grad Norm': norm.item()})
        
    train_loss /= len(train_dl.dataset)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0.0; e_rank = 0.0
    with torch.no_grad():
        p_bar = tqdm(val_dl, desc=f"Validation Epoch {epoch+1}/{epochs}")
        for batch in p_bar:
            videos = batch['video'].to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
                x_rec = model(videos)
                mse_loss = criterion(x_rec, videos)
            val_loss += mse_loss.item() * videos.size(0)
            e_rank += model.effective_rank.item() * videos.size(0)
            p_bar.set_postfix({'Recon': mse_loss.item(), 'EffRank': model.effective_rank.item()})
            
            
    val_loss /= len(val_dl.dataset)
    val_losses.append(val_loss)
    e_rank = e_rank / len(val_dl.dataset)
    e_ranks.append(e_rank)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Effective Rank: {e_rank:.2f}")

    # Saving model, reconstructions, losses
    torch.save(model.state_dict(), f"{output_dir}/MLAE.pth")
    plot_recons(model, val_ds, output_dir)
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(range(1, epoch + 2), train_losses, label='Training', color='tab:blue')
    ax1.plot(range(1, epoch + 2), val_losses, label='Validation', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, epoch + 2), e_ranks, label='EffRank', color='tab:green')
    ax2.set_ylabel('Effective Rank', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')
    
    plt.title('Losses over Epochs')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/losses.png", bbox_inches="tight")
    plt.close()