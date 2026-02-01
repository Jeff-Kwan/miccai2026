import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datahandling.EchoDynaDataset import load_echonet_dynamic_datasets
from models.MotionLatentPerceiver import MotionLatentPerceiver
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%Y_%m_%d")
timestamp = datetime.now().strftime("%H_%M")
output_dir = f"results/{date}/{timestamp}_MoLP"
os.makedirs(output_dir, exist_ok=True)

# Training Parameters
epochs = 100
batch_size = 16
learning_rate = 1e-4
weight_decay = 2e-2
masking = 0.75
max_frames = 64


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
    model.set_masking(False)
    with torch.no_grad():
        batch_device = batch.to(device)
        recon_batch = model(batch_device)[0]
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

    # Then pick a video to recon
    vid_idx = idxs[0]
    sample = val_ds[int(vid_idx)]
    video = sample['video']              # [C, T, H, W]
    video_batch = video.unsqueeze(0).to(device)  # [1, C, T, H, W]
    fps = sample.get("metadata", {}).get("FPS", 50)
    duration = 1.0 / float(fps)

    model.eval()
    model.set_masking(False)
    with torch.no_grad():
        recon_batch = model(video_batch)[0]
        recon_batch = recon_batch.cpu()

    video = video.cpu()
    recon = recon_batch[0]  # [C, T, H, W]

    # helper: tensor [C,H,W] in [-1,1] -> uint8 image
    def to_uint8(img_t):
        img_t = (img_t + 1.0) / 2.0
        img_t = img_t.clamp(0.0, 1.0)
        arr = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr

    T = video.shape[1]
    gif_frames = []

    for t in range(T):
        orig_np = to_uint8(video[:, t])
        recon_np = to_uint8(recon[:, t])

        fig, axs = plt.subplots(2, 1, figsize=(4, 6))
        for ax in axs:
            ax.axis("off")

        if orig_np.ndim == 2:
            axs[0].imshow(orig_np, cmap="gray")
            axs[1].imshow(recon_np, cmap="gray")
        else:
            axs[0].imshow(orig_np)
            axs[1].imshow(recon_np)

        axs[0].set_title("Original", fontsize=10)
        axs[1].set_title("Reconstruction", fontsize=10)
        plt.tight_layout()

        # Render matplotlib figure → numpy
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        gif_frames.append(frame)

        plt.close(fig)

    gif_path = os.path.join(output_dir, "recon_full_video.gif")
    imageio.mimsave(gif_path, gif_frames, duration=duration)


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
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(
    "data/echodyna/FileList.csv",
    "data/echodyna/Videos",
    "data/echodyna/VolumeTracings.csv",
    load_video=True)

def collate_fn(batch, blur=False):
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
    if blur:
        videos = median_blur(videos)
    return {'video': videos}


train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, blur=False),
                      num_workers=30, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, blur=False), 
                    num_workers=12)


# Model
model = MotionLatentPerceiver(in_c=3, out_c=3, init_c=8, latent=256, 
                              enc_layers=2, t_layers=12, t_heads=4, t_latents=4, 
                            dec_layers=2, levels=4, 
                            motion_dim=2,   # 2 templates
                            masking_ratio=0.75, skips=False)
model = model.to(device)
print(f"Initialized MoLP with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# Training 
train_losses = []; val_losses = []
for epoch in range(epochs):
    model.train()
    model.set_masking(True)
    train_loss = 0.0
    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in p_bar:
        videos = batch['video'].to(device, non_blocking=True)  # [B, C, T, H, W]
        
        optimizer.zero_grad()
        x_rec, = model(videos)
        
        recon_loss = criterion(x_rec, videos)
        loss = recon_loss
        
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += recon_loss.item() * videos.size(0)
        p_bar.set_postfix({'Recon': recon_loss.item(), 'Grad Norm': norm.item()})
        
        
    train_loss /= len(train_dl.dataset)
    train_losses.append(train_loss)
    
    model.eval()
    model.set_masking(False)
    val_loss = 0.0
    with torch.no_grad():
        p_bar = tqdm(val_dl, desc=f"Validation Epoch {epoch+1}/{epochs}")
        for batch in p_bar:
            videos = batch['video'].to(device, non_blocking=True)
            videos = median_blur(videos)
            x_rec = model(videos)[0]
            
            mse_loss = criterion(x_rec, videos)
            val_loss += mse_loss.item() * videos.size(0)
            p_bar.set_postfix({'MSE Loss': mse_loss.item()})
            
    val_loss /= len(val_dl.dataset)
    val_losses.append(val_loss)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Saving model, reconstructions, losses
    torch.save(model.state_dict(), f"{output_dir}/MoLP.pth")
    plot_recons(model, val_ds, output_dir)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epoch + 2), train_losses, label='Training')
    plt.plot(range(1, epoch + 2), val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Losses over Epochs')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"{output_dir}/losses.png", bbox_inches="tight")
    plt.close()