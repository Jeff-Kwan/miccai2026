import torch 
from datahandling.EchoDynaDataset import load_echonet_dynamic_datasets
from models.MotionLatentTransformer import MotionLatentAE
import os
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import imageio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_01_31/18_22_MLT"

model = MotionLatentAE(in_c=3, init_c=8, out_c=3, latent=256, 
                           enc_layers=2, t_layers=8, t_heads=4, t_latents=8,
                            dec_layers=2, levels=4, skips=False)
model = model.to(device)
model.load_state_dict(torch.load(os.path.join(load_dir, "MLT.pth"), map_location=device))
model = model.to(device)
model.eval()

train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(
    "data/echodyna/FileList.csv",
    "data/echodyna/Videos",
    "data/echodyna/VolumeTracings.csv",
    load_video=True)


idx = random.randint(0, len(val_ds) - 1)
video = val_ds[idx]['video'].to(device).unsqueeze(0)  # [1, C, T, H, W]

# Reconstruction
_, C, T, H, W = video.shape
with torch.inference_mode():
    if T <= 64:
        reconstruction, centroid = model(video)
        z_motion = model.z_motion.cpu().numpy().squeeze()
    else:
        # Process long videos in chunks
        chunk_size = 64
        reconstruction_chunks = []
        centroid_chunks = []
        z_motion_chunks = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            video_chunk = video[:, :, start:end, :, :]
            rec_chunk, cent_chunk = model(video_chunk)
            reconstruction_chunks.append(rec_chunk)
            centroid_chunks.append(cent_chunk)
            z_motion_chunks.append(model.z_motion.squeeze().cpu().numpy())
        reconstruction = torch.cat(reconstruction_chunks, dim=2)
        centroid = torch.cat(centroid_chunks, dim=2)
        z_motion = np.concatenate(z_motion_chunks, axis=1).squeeze()

output_dir = os.path.join(load_dir, "reconstructions")
os.makedirs(output_dir, exist_ok=True)

# Get FPS from dataset metadata
fps = val_ds[idx]["metadata"]["FPS"]
duration = 1.0 / fps  # seconds per frame for GIF

# Prepare tensors
video = video.squeeze(0).permute(1, 2, 3, 0)                 # [T, H, W, C]
reconstruction = reconstruction.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]
centroid = centroid.squeeze(0).permute(1, 2, 3, 0)          # [T, H, W, C]
video = (video + 1).clip(0, 1) * 255
reconstruction = (reconstruction + 1).clip(0, 1) * 255
centroid = (centroid + 1).clip(0, 1) * 255

video = video.cpu().numpy().astype(np.uint8)
reconstruction = reconstruction.cpu().numpy().astype(np.uint8)
centroid = centroid.cpu().numpy().astype(np.uint8)

T, H, W, C = video.shape
frames = []

for t in range(T):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(video[t])
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(reconstruction[t])
    axs[1].set_title("Reconstruction")
    axs[1].axis("off")

    axs[2].imshow(centroid[t])
    axs[2].set_title("Centroid")
    axs[2].axis("off")

    plt.tight_layout()

    # Render figure to numpy array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frames.append(frame)

    plt.close(fig)

# Save GIF using dataset FPS
gif_path = os.path.join(output_dir, f"{idx}-reconstruction.gif")
imageio.mimsave(gif_path, frames, duration=duration)


def plot_colormap_trajectory(x, y, title, xlabel, ylabel, save_path,
                             cmap="coolwarm"):
    """
    x, y: arrays of shape (T,)
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    t = np.linspace(0, 1, len(x))  # normalized time

    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(t)
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], color="blue", label="t=0", zorder=3)
    ax.scatter(x[-1], y[-1], color="red", label="t=T", zorder=3)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Normalized time")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

plot_colormap_trajectory(
    z_motion[0, :],
    z_motion[1, :],
    title="z_motion Trajectory",
    xlabel="D1",
    ylabel="D2",
    save_path=os.path.join(output_dir, f"{idx}-z_motion_trajectory.png"),
    cmap="coolwarm"
)