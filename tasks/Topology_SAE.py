# =========================
# Persistent homology (PH)
# =========================
# pip install ripser persim scikit-learn
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch 
from models.SplineAutoEncoder import SplineAutoEncoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from math import ceil
from persim import plot_diagrams
from sklearn.decomposition import PCA
from utils.filters import highpass, SavGolFilterTime
from dreimac import CircularCoords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_17/12_03_SAE"
ph_dir = os.path.join(load_dir, "topology")
os.makedirs(ph_dir, exist_ok=True)


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


idx = 998#random.randint(0, len(val_ds) - 1)
video = val_ds[idx]['video'].to(device).unsqueeze(0)
video = video * 2 - 1  # [0,1] → [-1,1]
timestamps = val_ds[idx]['timestamps'].unsqueeze(0).to(device)
frames_idx = val_ds[idx]['masks']['frame_indices']
gt_es, gt_ed = frames_idx
t0 = timestamps.min(); t1 = timestamps.max()    # 100 fps
# t_dense = torch.linspace(t0, t1, steps=int(200*(t1-t0)), device=device).unsqueeze(0)    


# Reconstruction
B, T, C, H, W = video.shape
with torch.inference_mode():
    with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
        z = model.encode(video)
        z_spline = model.spline_fit_and_eval(z, timestamps, timestamps).squeeze(0)

# Filters
z = (z - z.mean(dim=1, keepdim=True)).squeeze(0) # Remove static component
z_spline = (z_spline - z_spline.mean(dim=0, keepdim=True)) # Remove static component
z_original = z.clone().detach().cpu().numpy()
# keep 2 z_original eigenvectors
pca = PCA(n_components=2)
z_original = pca.fit_transform(z_original)

# Smoothing
# savgol = SavGolFilterTime(window_length=11, polyorder=3)
# z = savgol(z)
fps = val_ds[idx]["metadata"]["FPS"]
z_spline = highpass(z_spline, fs=fps, cutoff=0.5, order=2)

# ---- Choose what you want to analyze as a point cloud ----
Z_np = z_spline.detach().float().cpu().numpy()            # [T, D]
timestamps = timestamps.squeeze().cpu().numpy()  # [T]

# PCA and keep top k components
pca = PCA(n_components=0.99)
Z_np = pca.fit_transform(Z_np)  # [T, k]
print(f"Original latent dim: {z.shape[-1]}, after PCA: {Z_np.shape[1]}")

cc = CircularCoords(Z_np, n_landmarks=len(Z_np))
dgms = cc.dgms_

# Print 5 largest persistences of H1
if len(dgms) > 1:
    h1_dgm = dgms[1]
    persistences = h1_dgm[:, 1] - h1_dgm[:, 0]
    top_indices = np.argsort(persistences)[-5:][::-1]
    print("5 largest H1 persistences:")
    for index in top_indices:
        birth, death = h1_dgm[index]
        persistence = death - birth
        print(f"  [ {birth:.3f}, {death:.3f} ) - persistence: {persistence:.3f}")

# Phase
circular_coordinates = cc.get_coordinates(perc=0.95)

# Normalize to ed phase = 0
ed_phase = circular_coordinates[gt_ed]
circular_coordinates = (circular_coordinates - ed_phase + 2 * np.pi) % (2 * np.pi)

fig, ax = plt.subplots()
scatter = ax.scatter(z_original[:,0], z_original[:,1], c=circular_coordinates, s=6, cmap="viridis")
ax.set_title("Phase Color Plot of Z - PC1 & PC2")
ax.axis("off")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Phase')
cbar.ax.set_yticks([0, np.pi, 2*np.pi])
cbar.ax.set_yticklabels(['0', 'π', '2π'])
plt.savefig(os.path.join(ph_dir, f"{idx}-phase_color.png"), dpi=200)

# Phase vs time plot
circular_coordinates = np.sin(circular_coordinates)

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(timestamps, circular_coordinates, color='blue', s=3)
for f in frames_idx.squeeze().cpu().numpy():
    ax.scatter(timestamps[f], circular_coordinates[f], color='red', marker="x", s=100, label="ES/ED", zorder=3)
for phase_val in [-1, 0, 1]:#[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
    ax.axhline(phase_val, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Sin Phase")
# ax.set_yticks([0, np.pi, 2*np.pi])
# ax.set_yticklabels(['0', 'π', '2π'])
plt.savefig(os.path.join(ph_dir, f"{idx}-phase_time.png"), dpi=200)

# Norm of z_motion vs time
z_motion_norm = np.linalg.norm(z.clone().detach().cpu().numpy(), axis=1).squeeze()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(timestamps, z_motion_norm, color='green', linewidth=1)
for f in frames_idx.squeeze().cpu().numpy():
    ax.scatter(timestamps[f], z_motion_norm[f], color='red', marker="x", s=100, label="ES/ED", zorder=3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Norm of z_motion")
plt.savefig(os.path.join(ph_dir, f"{idx}-znorm_time.png"), dpi=200)

plt.close()