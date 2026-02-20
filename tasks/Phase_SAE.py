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
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter, find_peaks
from utils.topology import cohomology_circular_coords, plot_phase_and_z, plot_phase_and_time, \
        plot_znorm_and_time, plot_phase_major_axis, laplacian_phase, highpass_filter, detrend, \
        preprocess_to_tangent_space, find_phase_major_axis
from tasks.Compute_EDES import EDES_via_Phase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_19/16_20_SAE"
out_dir = os.path.join(load_dir, "topology")
os.makedirs(out_dir, exist_ok=True)


config = json.load(open("config/SAE.json", "r"))
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


idx = 366#random.randint(0, len(test_ds) - 1)
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

# EDES_via_Phase(z, z_spline, timestamps, fps, gt_ed, gt_es)
# exit()

ed_err, es_err = EDES_via_Phase(z, z_spline, timestamps, fps, gt_ed, gt_es)
print(f"ED error: {ed_err:.2f} frames, ES error: {es_err:.2f} frames")

# phase, dgms = cohomology_circular_coords(
#     z_spline, fps=fps,
#     print_dgms_summary=True) 
phase, evals, evecs = laplacian_phase(z_spline)

z_spline = z_spline[::dense_factor]
phase = phase[::dense_factor]
t_dense = t_dense[::dense_factor]

z_proj = z_spline @ find_phase_major_axis(z_spline, phase)
z_proj = detrend(z_proj, type='linear')
group1 = find_peaks(z_proj, prominence=0.2*(np.max(z_proj)-np.min(z_proj)), distance=5)[0]
group2 = find_peaks(-z_proj, prominence=0.2*(np.max(z_proj)-np.min(z_proj)), distance=5)[0]
peaks = np.concatenate([group1, group2])
print(peaks)

plot_phase_major_axis(z_spline, t_dense, phase, out_dir, idx, frames_idx=frames_idx, peaks=peaks)

# PCA for plotting
pca = PCA(n_components=3)
# z_spline = highpass_filter(z_spline, fs=fps, cutoff=0.5, order=4, axis=0)
# z_spline = preprocess_to_tangent_space(z_spline, pca=True)
z_spline_3d = pca.fit_transform(z_spline)

# plot_phase_and_z(z_spline_3d, phase, out_dir, idx, dim="2d", gt_ed=gt_ed, frames_idx=frames_idx)
plot_phase_and_z(z_spline_3d, phase, out_dir, idx, dim="3d", gt_ed=gt_ed, frames_idx=frames_idx)
plot_phase_and_time(phase, t_dense, out_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx, differentiate=0)
# plot_phase_and_time(phase, t_dense, out_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx, differentiate=1)
plot_znorm_and_time(z_spline_3d, t_dense, out_dir, idx, frames_idx=frames_idx)