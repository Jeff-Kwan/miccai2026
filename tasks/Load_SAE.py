import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch 
from models.SplineAutoEncoder import SplineAutoEncoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
import os
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import numpy as np
import imageio
import json
from math import ceil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_15/15_04_SAE"
output_dir = os.path.join(load_dir, "reconstructions")
os.makedirs(output_dir, exist_ok=True)


config = json.load(open("config/SAE.json", "r"))
mcfg = config["model"]
model = SplineAutoEncoder(
    latent=mcfg["latent"],
    in_dim=mcfg.get("in_dim", 3),
    out_dim=mcfg.get("out_dim", None),
    n_ctrl=ceil(config["training"]["max_frames"]//3)+3,
    degree=3,
    lam=1e-3,
).to(device)
model.load_state_dict(torch.load(os.path.join(load_dir, "SAE.pth"), map_location=device))
autocast = config["training"].get("autocast", False)

train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=True)


idx = 1113#random.randint(0, len(val_ds) - 1)
video = val_ds[idx]['video'].to(device).unsqueeze(0)
video = video * 2 - 1  # [0,1] → [-1,1]
timestamps = val_ds[idx]['timestamps'].unsqueeze(0).to(device)
frames_idx = val_ds[idx]['masks']['frame_indices']


# Reconstruction
B, T, C, H, W = video.shape
with torch.inference_mode():
    with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
        z = model.encode(video)
        z = model.spline_fit_and_eval(z, timestamps, timestamps)
        reconstruction = model.decode(z, H, W)
        z_motion = z - z.mean(dim=1, keepdim=True) # Remove static component

# Get FPS from dataset metadata
fps = val_ds[idx]["metadata"]["FPS"]
duration = 1.0 / fps  # seconds per frame for GIF

# Prepare tensors
video = video.squeeze(0).permute(0, 2, 3, 1)
reconstruction = reconstruction.squeeze(0).permute(0, 2, 3, 1)
video = (video/2 + 0.5).clip(0, 1) * 255
reconstruction = (reconstruction/2 + 0.5).clip(0, 1) * 255

video = video.cpu().numpy().astype(np.uint8)
reconstruction = reconstruction.cpu().numpy().astype(np.uint8)
z_motion = z_motion.squeeze(0).float().cpu().numpy()  # shape [T, latent]

# Print participation ratio of z_motion
cov = np.cov(z_motion, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
total_variance = eigenvalues.sum()
participation_ratio = (eigenvalues.sum() ** 2) / (np.sum(eigenvalues ** 2) + 1e-10)
print(f"Participation Ratio of z_motion: {participation_ratio:.2f}")

# PCA for top 2 components
top2_indices = np.argsort(eigenvalues)[-2:]
top2_eigenvectors = eigenvectors[:, top2_indices]
z_motion = z_motion @ top2_eigenvectors  # shape [T, 2]


T, H, W, C = video.shape
frames = []

for t in range(T):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(video[t])
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(reconstruction[t])
    axs[1].set_title("Reconstruction")
    axs[1].axis("off")
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


def plot_colormap_trajectory(x, y, t, title, xlabel, ylabel, save_path, frames_idx,
                             cmap="coolwarm"):
    """
    x, y: arrays of shape (T,)
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(t)
    lc.set_linewidth(2)

    # Layout: stack trajectory, D1 vs time, D2 vs time vertically
    fig = plt.figure(figsize=(6, 12))
    gs0 = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)

    # Wrapper to keep later indexing (gs[1,0], gs[1,1]) working while stacking vertically.
    class GSWrapper:
        def __init__(self, gs):
            self.gs = gs
        def __getitem__(self, key):
            # map (1,1) used later to the third row so the two component plots end up stacked
            if key == (1, 1):
                return self.gs[2, 0]
            return self.gs[key]

    gs = GSWrapper(gs0)

    ax_traj = fig.add_subplot(gs[0, :])
    ax_traj.add_collection(lc)
    ax_traj.scatter(x[0], y[0], color="blue", label="t=0", zorder=3)
    ax_traj.scatter(x[-1], y[-1], color="red", label="t=T", zorder=3)
    for f in frames_idx:
        ax_traj.scatter(x[f], y[f], color="green", marker="x", s=100, label="ES/ED", zorder=3)

    pad_x = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 1.0
    pad_y = (y.max() - y.min()) * 0.05 if y.max() > y.min() else 1.0
    ax_traj.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax_traj.set_ylim(y.min() - pad_y, y.max() + pad_y)
    ax_traj.set_title(title)
    ax_traj.set_xlabel(xlabel)
    ax_traj.set_ylabel(ylabel)
    ax_traj.legend()
    cbar = fig.colorbar(lc, ax=ax_traj, fraction=0.046, pad=0.04)
    cbar.set_label("Time")

    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.scatter(t, x, color="black", s=1)
    for f in frames_idx:
        ax_x.scatter(t[f], x[f], color="green", marker="x", s=100, label="ES/ED", zorder=3)
    ax_x.set_xlabel("Time (sec)")
    ax_x.set_ylabel("D1")
    ax_x.grid(True, linestyle=":", alpha=0.6)

    ax_y = fig.add_subplot(gs[1, 1])
    ax_y.scatter(t, y, color="black", s=1)
    for f in frames_idx:
        ax_y.scatter(t[f], y[f], color="green", marker="x", s=100, label="ES/ED", zorder=3)
    ax_y.set_xlabel("Time (sec)")
    ax_y.set_ylabel("D2")
    ax_y.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

plot_colormap_trajectory(
    z_motion[:, 0],
    z_motion[:, 1],
    timestamps.squeeze().cpu().numpy(),
    title="z_motion Trajectory",
    xlabel="D1",
    ylabel="D2",
    save_path=os.path.join(output_dir, f"{idx}-z_motion_trajectory.png"),
    cmap="coolwarm",
    frames_idx=frames_idx)