# =========================
# Loop phase analysis
# =========================
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch 
from models.SplineAutoEncoder import SplineAutoEncoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
import os
import json
from math import ceil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from scipy.signal import savgol_filter, find_peaks
from utils.topology import cohomology_circular_coords, plot_phase_and_z, plot_phase_and_time, \
        plot_znorm_and_time, plot_phase_major_axis, laplacian_phase, highpass_filter, detrend, \
        find_phase_major_axis, preprocess_z, project_to_phase_plane, von_mises_kernel_smoother
from tasks.Compute_EDES import EDES_via_Phase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_22/18_28_SAE"
out_dir = os.path.join(load_dir, "topology")
os.makedirs(out_dir, exist_ok=True)


config = json.load(open(os.path.join(load_dir, "config.json"), "r"))
mcfg = config["model"]
model = SplineAutoEncoder(
    latent=config["model"]["latent"],
    in_dim=config["model"].get("in_dim", 3),
    out_dim=config["model"].get("out_dim", None),
    n_ctrl_params=config["model"]["n_ctrl_params"],
    degree=config["model"]["degree"],
    lam=config["model"]["lam"],
).to(device)
model.load_state_dict(torch.load(os.path.join(load_dir, "SAE.pth"), map_location=device))
autocast = config["training"].get("autocast", False)

train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=True)


idx = random.randint(0, len(test_ds) - 1)
video = test_ds[idx]['video'].to(device).unsqueeze(0)
video = video * 2 - 1  # [0,1] → [-1,1]
timestamps = test_ds[idx]['timestamps'].unsqueeze(0).to(device)
frames_idx = test_ds[idx]['masks']['frame_indices']
gt_es = frames_idx[0].item(); gt_ed = frames_idx[1].item()
fps = test_ds[idx]["metadata"]["FPS"]

t0 = timestamps.min(); t1 = timestamps.max(); T = timestamps.shape[1]
dense_factor = 1
t_dense = torch.linspace(t0, t1, (T-1)*dense_factor+1, device=device).unsqueeze(0)
print(f"Video has {T} frames, from {t0:.2f}s to {t1:.2f}s at {fps:.2f} FPS. Dense timestamps has shape {t_dense.shape}.")

# Reconstruction
B, T, C, H, W = video.shape
with torch.inference_mode():
    with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
        z = model.encode(video)
        z_spline = model.spline_fit_and_eval(z, timestamps, t_dense)
z = (z - z.mean(dim=1, keepdim=True)).squeeze(0).cpu().numpy()
z_spline = (z_spline - z_spline.mean(dim=1, keepdim=True)).squeeze(0).cpu().numpy()
timestamps = timestamps.squeeze(0).cpu().numpy()
t_dense = t_dense.squeeze(0).cpu().numpy()  # [T_dense]

# Participation ratios
def participation_ratio(z):
    cov = np.cov(z, rowvar=False)
    evals, _ = np.linalg.eigh(cov)
    evals = np.sort(evals)[::-1]
    pr = (evals.sum() ** 2) / (np.square(evals).sum() + 1e-8)
    return pr, evals
print("Participation Ratio (raw):", participation_ratio(z)[0])
print("Participation Ratio (spline):", participation_ratio(z_spline)[0])

# Detrend
z = detrend(z, axis=0, type='linear')
z_spline = detrend(z_spline, axis=0, type='linear')

# phase, dgms = cohomology_circular_coords(
#     z_spline,
#     print_dgms_summary=True) 
phase, evals, evecs = laplacian_phase(z_spline)

z_spline = z_spline[::dense_factor]
phase = phase[::dense_factor]
t_dense = t_dense[::dense_factor]

z_proj = z @ find_phase_major_axis(z, phase)
z_proj = detrend(z_proj, type='linear')
group1 = find_peaks(z_proj, prominence=0.2*(np.max(z_proj)-np.min(z_proj)), distance=5)[0]
group2 = find_peaks(-z_proj, prominence=0.2*(np.max(z_proj)-np.min(z_proj)), distance=5)[0]
peaks = np.concatenate([group1, group2])
print(peaks)

plot_phase_major_axis(z, t_dense, phase, out_dir, idx, frames_idx=frames_idx, peaks=peaks)


grid, mu = von_mises_kernel_smoother(z, phase, n_grid=512, kappa=30)
z_phase_plane, pp_basis = project_to_phase_plane(z, phase)
plt.scatter(z_phase_plane[:, 0], z_phase_plane[:, 1], c=phase, cmap='hsv', s=5)
plt.scatter(z_phase_plane[frames_idx, 0], z_phase_plane[frames_idx, 1], c='black', s=50, label='ES/ED frames')
plt.colorbar(label='Phase')
plt.title(f"Processed Z on Phase Plane")
plt.xlabel("Phase Plane X")
plt.ylabel("Phase Plane Y")
plt.savefig(os.path.join(out_dir, f"{idx}-phase_plane-Processed.png"), dpi=200)
plt.close()

z_phase_plane = z @ pp_basis
mu_plane = mu @ pp_basis
plt.scatter(z_phase_plane[:, 0], z_phase_plane[:, 1], c=phase, cmap='hsv', s=5)
plt.plot(mu_plane[:, 0], mu_plane[:, 1], c='black', linewidth=2, label='Smoothed Trajectory')
plt.scatter(z_phase_plane[frames_idx, 0], z_phase_plane[frames_idx, 1], c='red', marker='x', s=50, label='ES/ED frames')
plt.colorbar(label='Phase')
plt.title(f"Actual Z on Phase Plane")
plt.xlabel("Phase Plane X")
plt.ylabel("Phase Plane Y")
plt.savefig(os.path.join(out_dir, f"{idx}-phase_plane-Actual.png"), dpi=200)
plt.close()

# PCA for plotting
pca = PCA(n_components=3)
z_3d = pca.fit_transform(z)
mu_3d = pca.transform(mu)

# plot_phase_and_z(z_3d, phase, out_dir, idx, dim="2d", gt_ed=gt_ed, frames_idx=frames_idx, mu=mu_3d)
plot_phase_and_z(z_3d, phase, out_dir, idx, dim="3d", gt_ed=gt_ed, frames_idx=frames_idx, mu=mu_3d)
plot_phase_and_time(phase, t_dense, out_dir, idx, sine=False, gt_ed=gt_ed, frames_idx=frames_idx, differentiate=0)
# plot_phase_and_time(phase, t_dense, out_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx, differentiate=1)
# plot_znorm_and_time(mu, t_dense, out_dir, idx, frames_idx=frames_idx)

ed_phase = phase[gt_ed]; es_phase = phase[gt_es]
# nearest neighbor grid idx for each event
ed_grid_idx = np.argmin(np.abs(grid - ed_phase))
es_grid_idx = np.argmin(np.abs(grid - es_phase))
mu_norm = np.linalg.norm(mu, axis=-1)
plt.scatter(grid, mu_norm, color='black', s=10)
plt.scatter(grid[ed_grid_idx], mu_norm[ed_grid_idx], color='red', s=50, label='ED')
plt.scatter(grid[es_grid_idx], mu_norm[es_grid_idx], color='blue', s=50, label='ES')
plt.title("Smoothed Trajectory Norm vs Phase")
plt.xlabel("Phase")
plt.ylabel("Norm of Smoothed Trajectory")
plt.legend()
plt.savefig(os.path.join(out_dir, f"{idx}-mu_norm_vs_phase.png"), dpi=200)
plt.close()