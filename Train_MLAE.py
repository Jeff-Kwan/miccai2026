import torch 
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datahandling.EchoDynaDataset import load_echonet_dynamic_datasets
from models.MotionLatentAE import MotionLatentAE
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
batch_size = 16
learning_rate = 3e-4
weight_decay = 1e-3
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
    with torch.no_grad():
        batch_device = batch.to(device)
        recon_batch, _ = model(batch_device)
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
        recon_clip_batch, _ = model(clip_batch)
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
    plt.close(fig)



# Dataset
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(
    "data/echodyna/FileList.csv",
    "data/echodyna/Videos",
    "data/echodyna/VolumeTracings.csv",
    load_video=True)

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

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, 
                      num_workers=10, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=6)


# Model
model = MotionLatentAE(in_c=3, out_c=3, latent=512, enc_layers=4, dec_layers=2, levels=6)
model = model.to(device)
print(f"Initialized MLAE with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def effective_rank(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Differentiable 'rank-like' scalar in [1, n] (approximately),
    based on entropy of singular values.

    Returns effective rank (not log-rank). To make it a penalty, minimize it.
    Supports A shaped (n,n) or batched (..., n, n).
    """
    s = torch.linalg.svdvals(A)                      # (..., n)
    s_sum = s.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = (s / s_sum).clamp_min(eps)                   # (..., n)
    H = -(p * p.log()).sum(dim=-1)                   # (...)
    return H.exp()                                   # (...)


# Training 
train_losses = []; val_losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in p_bar:
        videos = batch['video'].to(device, non_blocking=True)  # [B, C, T, H, W]
        
        optimizer.zero_grad()
        x_rec, x_centroid = model(videos)
        
        mse_loss = criterion(x_rec, videos)
        frechet_loss = criterion(x_centroid, videos)
        loss = mse_loss + frechet_loss / videos.size(2)
        
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += mse_loss.item() * videos.size(0)
        p_bar.set_postfix({'MSE Loss': mse_loss.item(), 'Frechet Loss': frechet_loss.item(), 'Grad Norm': norm.item()})
        
    train_loss /= len(train_dl.dataset)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        p_bar = tqdm(val_dl, desc=f"Validation Epoch {epoch+1}/{epochs}")
        for batch in p_bar:
            videos = batch['video'].to(device, non_blocking=True)
            x_rec, _ = model(videos)
            
            mse_loss = criterion(x_rec, videos)
            val_loss += mse_loss.item() * videos.size(0)
            p_bar.set_postfix({'MSE Loss': mse_loss.item()})
            
            
    val_loss /= len(val_dl.dataset)
    val_losses.append(val_loss)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Saving model, reconstructions, losses
    torch.save(model.state_dict(), f"{output_dir}/MLAE.pth")
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