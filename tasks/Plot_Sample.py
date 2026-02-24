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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import detrend, savgol_filter
from utils.topology import laplacian_phase, project_to_major_axis, von_mises_kernel_smoother
from tasks.Compute_EDES import find_peaks_sentinel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_23/15_20_SAE"
out_dir = os.path.join(load_dir, "sample")
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

_, _, test_ds = load_echonet_dynamic_datasets(get_mask=True)

idx = 20
video = test_ds[idx]['video'].to(device).unsqueeze(0)
video = video * 2 - 1  # [0,1] → [-1,1]
timestamps = test_ds[idx]['timestamps'].unsqueeze(0).to(device)
gt_ed = int(test_ds[idx]['ED']); gt_es = int(test_ds[idx]['ES']); fps = float(test_ds[idx]['fps'])
B, T, C, H, W = video.shape
with torch.inference_mode():
    z = model.encode(video)
    recon = model.decode(z, H=H, W=W)
z = z.squeeze().cpu().numpy()
timestamps = timestamps.squeeze().cpu().numpy()
recon = recon.squeeze().permute(0, 2, 3, 1).cpu().numpy()
video = video.squeeze().permute(0, 2, 3, 1).cpu().numpy()


plt.rcParams.update({
    "font.family": "serif",          # or "Times New Roman"
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
})

# Save original ED ES frames
ed_frame = video[gt_ed]
es_frame = video[gt_es]
plt.imsave(os.path.join(out_dir, f"{idx}-ED.png"), (ed_frame/2 + 0.5).clip(0, 1))
plt.imsave(os.path.join(out_dir, f"{idx}-ES.png"), (es_frame/2 + 0.5).clip(0, 1))

# Save reconstructed ED ES frames
recon_ed_frame = recon[gt_ed]
recon_es_frame = recon[gt_es]
plt.imsave(os.path.join(out_dir, f"{idx}-recon-ED.png"), (recon_ed_frame/2 + 0.5).clip(0, 1))
plt.imsave(os.path.join(out_dir, f"{idx}-recon-ES.png"), (recon_es_frame/2 + 0.5).clip(0, 1))

# Assign Phase and Processing
EDES_global = np.load("tasks/EDES_axis.npy")
z = detrend(z, axis=0, type='linear')
phase = laplacian_phase(z)[0]
# phase = phase / (np.max(phase)-np.min(phase)) * (2*np.pi+0.01) # Full colorbar
z_proj = savgol_filter(project_to_major_axis(z, phase, axis=EDES_global), window_length=11, polyorder=3, axis=0)
g1, g2 = find_peaks_sentinel(z_proj, p=0.2, d=5)
peaks = np.concatenate([g1, g2])

# Plot 1D projection colored by phase
# plt.scatter(timestamps, z_proj, c=phase, cmap='hsv', s=5)
# cbar = plt.colorbar(label='Circular Coordinate Phase')
# cbar.ax.set_yticks([-np.pi+0.07, 0, np.pi-0.07])    # Fuller colorbar
# cbar.ax.set_yticklabels(['-π', '0', 'π'])
# plt.xlabel('Time (s)')
# plt.ylabel('Projection')
# plt.axvline(x=timestamps[gt_ed], color='blue', linestyle='--', label='Ground Truth ED')
# plt.axvline(x=timestamps[gt_es], color='green', linestyle='--', label='Ground Truth ES')
# for p in peaks:
#     plt.scatter(timestamps[p], z_proj[p], color='black', marker="x", s=120, zorder=3)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, f"{idx}-phase_color_1d.png"), dpi=300)
# plt.clf()
# plt.close()




fig, ax = plt.subplots(figsize=(6, 3))
sc = ax.scatter(timestamps, z_proj, c=phase, cmap='hsv', s=8, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
# cbar.set_label('Phase')
cbar.set_ticks([-np.pi+0.05, 0, np.pi-0.05])
cbar.set_ticklabels(['−π', '0', 'π'])
ax.scatter(timestamps[gt_ed], z_proj[gt_ed], marker='x', s=100, linewidths=2.0, color='#C1121F', label='ED', zorder=5)
ax.scatter(timestamps[gt_es], z_proj[gt_es], marker='x', s=100, linewidths=2.0, color='#0057B8', label='ES', zorder=5)
ax.set_xlabel('Time (s)')
# ax.set_ylabel('1D Projection')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right', frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"{idx}-phase_color_1d.png"),dpi=300)
plt.close(fig)

exit()
grid, mu = von_mises_kernel_smoother(z, phase, n_grid=512, kappa=30)
z_phase_plane, pp_basis = project_to_phase_plane(z, phase)
plt.scatter(z_phase_plane[:, 0], z_phase_plane[:, 1], c=phase, cmap='hsv', s=5)
plt.scatter(z_phase_plane[gt_ed, 0], z_phase_plane[gt_ed, 1], c='red', s=50, label='ED frame')
plt.scatter(z_phase_plane[gt_es, 0], z_phase_plane[gt_es, 1], c='blue', s=50, label='ES frame')
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
plt.scatter(z_phase_plane[gt_ed, 0], z_phase_plane[gt_ed, 1], c='red', marker='x', s=50, label='ED frame')
plt.scatter(z_phase_plane[gt_es, 0], z_phase_plane[gt_es, 1], c='blue', marker='x', s=50, label='ES frame')
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

# plot_phase_and_z(z_3d, phase, out_dir, idx, dim="2d", gt_ed=gt_ed, gt_es=gt_es, mu=mu_3d)
plot_phase_and_z(z_3d, phase, out_dir, idx, dim="3d", gt_ed=gt_ed, gt_es=gt_es, mu=mu_3d)
plot_phase_and_time(phase, t_dense, out_dir, idx, sine=False, gt_ed=gt_ed, gt_es=gt_es, differentiate=0)
# plot_phase_and_time(phase, t_dense, out_dir, idx, gt_ed=gt_ed, gt_es=gt_es, differentiate=1)
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