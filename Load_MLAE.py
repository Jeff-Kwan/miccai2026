import torch 
from datahandling.EchoDynaDataset import load_echonet_dynamic_datasets
from models.MotionLatentAE2 import MotionLatentAE
import os
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import numpy as np
import imageio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_08/10_21_MLAE"
output_dir = os.path.join(load_dir, "reconstructions")
os.makedirs(output_dir, exist_ok=True)

model = MotionLatentAE(in_c=3, out_c=3, latent=256, enc_layers=4, 
                           dec_layers=2, levels=5, skips=False)
model = model.to(device)
model.load_state_dict(torch.load(os.path.join(load_dir, "MLAE.pth"), map_location=device))
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
    reconstruction = model(video)
    # z_motion = model.z_motion.cpu().numpy().squeeze()

v = model.v.cpu().numpy().squeeze()  # [T, latent]

# Compute effective rank of v
_, s, _ = np.linalg.svd(v, full_matrices=False)
e = s**2
p = e / e.sum()
H = -(p * np.log(p)).sum()
effective_rank = np.exp(H)
print(f"Effective rank of v: {effective_rank:.4f}")

# Plot the singular value spectrum up to effective rank
num_plot = min(max(int(4*effective_rank), 40), len(s))
plt.figure(figsize=(6, 4))
plt.plot(s[:num_plot], marker='o')
plt.title("Singular Value Spectrum of v")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.yscale("log")
plt.axvline(x=effective_rank, color='r', linestyle='--', label=f"Effective rank: {effective_rank:.2f}")
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend()
plt.savefig(os.path.join(output_dir, f"{idx}-v_singular_values.png"), dpi=300, bbox_inches="tight")
plt.close()

# z_motion = PC1, PC2
z_motion = v[:, :2]  # [T, 2]



# Get FPS from dataset metadata
fps = val_ds[idx]["metadata"]["FPS"]
duration = 1.0 / fps  # seconds per frame for GIF

# Prepare tensors
video = video.squeeze(0).permute(1, 2, 3, 0)                 # [T, H, W, C]
reconstruction = reconstruction.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]
video = (video + 1).clip(0, 1) * 255
reconstruction = (reconstruction + 1).clip(0, 1) * 255

video = video.cpu().numpy().astype(np.uint8)
reconstruction = reconstruction.cpu().numpy().astype(np.uint8)

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


def plot_colormap_trajectory(x, y, esed, title, xlabel, ylabel, save_path,
                             cmap="coolwarm"):
    """
    x, y: arrays of shape (T,)
    """
    es, ed = esed  # end-systole, end-diastole indices
    ed_idx = int(ed)
    es_idx = int(es)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    t = np.linspace(0, 1, len(x))  # normalized time

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

    ax_traj.plot(x[ed_idx], y[ed_idx], marker="x", color="orange", markersize=8, markeredgewidth=2, label="ED")
    ax_traj.plot(x[es_idx], y[es_idx], marker="x", color="green", markersize=8, markeredgewidth=2, label="ES")

    pad_x = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 1.0
    pad_y = (y.max() - y.min()) * 0.05 if y.max() > y.min() else 1.0
    ax_traj.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax_traj.set_ylim(y.min() - pad_y, y.max() + pad_y)
    ax_traj.set_title(title)
    ax_traj.set_xlabel(xlabel)
    ax_traj.set_ylabel(ylabel)
    ax_traj.legend()
    cbar = fig.colorbar(lc, ax=ax_traj, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized time")

    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.scatter(t, x, color="black", s=1)
    ax_x.set_xlabel("Normalized time")
    ax_x.set_ylabel("D1")
    ax_x.grid(True, linestyle=":", alpha=0.6)
    ax_x.plot(t[ed_idx], x[ed_idx], marker="x", color="orange", markersize=8, markeredgewidth=2)
    ax_x.plot(t[es_idx], x[es_idx], marker="x", color="green", markersize=8, markeredgewidth=2)

    ax_y = fig.add_subplot(gs[1, 1])
    ax_y.scatter(t, y, color="black", s=1)
    ax_y.set_xlabel("Normalized time")
    ax_y.set_ylabel("D2")
    ax_y.grid(True, linestyle=":", alpha=0.6)
    ax_y.plot(t[ed_idx], y[ed_idx], marker="x", color="orange", markersize=8, markeredgewidth=2)
    ax_y.plot(t[es_idx], y[es_idx], marker="x", color="green", markersize=8, markeredgewidth=2)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

plot_colormap_trajectory(
    z_motion[:, 0],
    z_motion[:, 1],
    esed=(val_ds[idx]['metadata']['ESV'], val_ds[idx]['metadata']['EDV']),
    title="z_motion Trajectory",
    xlabel="D1",
    ylabel="D2",
    save_path=os.path.join(output_dir, f"{idx}-z_motion_trajectory.png"),
    cmap="coolwarm")