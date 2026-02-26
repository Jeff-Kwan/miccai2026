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
from sklearn.manifold import SpectralEmbedding
from utils.topology import laplacian_phase, project_to_major_axis, von_mises_kernel_smoother, project_to_phase_plane
from tasks.Compute_EDES import find_peaks_sentinel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_24/14_57_SAE"
out_dir = os.path.join(load_dir, "more_samples")
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

idx = 822
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
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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


fig, ax = plt.subplots(figsize=(6, 3))
sc = ax.scatter(timestamps, z_proj, c=phase, cmap='hsv', s=8)
cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
cbar.set_ticks([-np.pi+0.05, 0, np.pi-0.05])
cbar.set_ticklabels(['−π', '0', 'π'])
ax.scatter(timestamps[gt_ed], z_proj[gt_ed], marker='x', s=100, linewidths=2.0, color='#C1121F', label='ED', zorder=5)
ax.scatter(timestamps[gt_es], z_proj[gt_es], marker='x', s=100, linewidths=2.0, color='#0057B8', label='ES', zorder=5)
ax.set_xlabel('Time (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{idx}-phase_color_1d.png"),dpi=300)
plt.clf()
plt.close()


pca = PCA(n_components=3)
z_3d = pca.fit_transform(z)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z_3d[:,0], z_3d[:,1], z_3d[:,2], c=phase, cmap='hsv', s=10, depthshade=False)
ax.scatter(z_3d[gt_ed,0], z_3d[gt_ed,1], z_3d[gt_ed,2], marker='x', s=200, linewidths=3, color='#C1121F', label='ED', depthshade=False)
ax.scatter(z_3d[gt_es,0], z_3d[gt_es,1], z_3d[gt_es,2], marker='x', s=200, linewidths=3, color='#0057B8', label='ES', depthshade=False)
ax.legend(loc='upper right', frameon=False)
# ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
plt.savefig(os.path.join(out_dir, f"{idx}-latent_3d.png"), dpi=300)
plt.clf()
plt.close()




grid, mu = von_mises_kernel_smoother(z, phase, kappa=20, n_grid=256)
mu_3d = pca.transform(mu)
dmu_3d = savgol_filter(mu_3d, deriv=1, window_length=11, polyorder=3, axis=0)
dmu_norm = dmu_3d / (np.linalg.norm(dmu_3d, axis=1, keepdims=True) + 1e-8)
step = 10
idxs = np.arange(0, len(mu_3d), step)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(z_3d[:,0], z_3d[:,1], z_3d[:,2], color='gray', linewidth=1.0, alpha=0.6)
ax.quiver(mu_3d[idxs,0], mu_3d[idxs,1], mu_3d[idxs,2], dmu_3d[idxs,0], dmu_3d[idxs,1], dmu_3d[idxs,2], length=0.05, normalize=True, color='orange', linewidth=2.0, arrow_length_ratio=0.6)
# ax.scatter(z_3d[gt_ed,0], z_3d[gt_ed,1], z_3d[gt_ed,2], marker='x', s=200, linewidths=3, color='#C1121F', label='ED', depthshade=False)
# ax.scatter(z_3d[gt_es,0], z_3d[gt_es,1], z_3d[gt_es,2], marker='x', s=200, linewidths=3, color='#0057B8', label='ES', depthshade=False)
# ax.legend(frameon=False)
# ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
plt.savefig(os.path.join(out_dir, f"{idx}-direction_flow_3d.png"), dpi=300)
plt.clf()
plt.close()


dz = savgol_filter(z, deriv=1, window_length=11, polyorder=3, axis=0)
dz = dz / np.linalg.norm(dz, axis=1, keepdims=True)
spectral_embedding = SpectralEmbedding(
        n_components=10, 
        n_neighbors=15, 
        affinity='nearest_neighbors',
        n_jobs=-1)
evecs = spectral_embedding.fit_transform(dz)
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(evecs[:,0], evecs[:,1], c=phase, cmap='hsv', s=10)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{idx}-spectral_embedding.png"), dpi=300)
plt.clf()
plt.close()

# plot spectrum?
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
W = spectral_embedding.affinity_matrix_   # kNN graph (sparse)
L = csgraph_laplacian(W, normed=True)
# How many eigenvalues to look at (more than n_components is useful)
m = min(30, L.shape[0] - 2)  # don't ask for >= N eigenpairs
evals, _ = eigsh(L, k=m, which="SM")      # smallest magnitude eigenvalues
evals = np.sort(np.real(evals))
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(np.arange(1, len(evals)+1), evals, marker="o", linewidth=1)
# ax.set_xlabel("Eigenvalue index")
# ax.set_ylabel("Eigenvalue")
# ax.set_title("Graph Laplacian Spectrum")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{idx}-laplacian_spectrum.png"), dpi=300)
plt.close(fig)