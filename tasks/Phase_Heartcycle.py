# =========================
# Phase Analysis on Heartcycle Dataset
# =========================
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch 
from models.SplineAutoEncoder1D import SplineAutoEncoder1D
from datahandling.HeartcycleDataset import HeartcycleDataset
import os
import json
from math import ceil
import random
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from utils.topology import cohomology_circular_coords, plot_phase_and_z, plot_phase_and_time, \
        plot_znorm_and_time, plot_phase_major_axis, laplacian_phase, highpass_filter, detrend, \
        preprocess_to_tangent_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_19/15_14_Heartcycle"
out_dir = os.path.join(load_dir, "phase_analysis")
os.makedirs(out_dir, exist_ok=True)


config = json.load(open("config/SAE_1D.json", "r"))
mcfg = config["model"]
model = SplineAutoEncoder1D(
    latent=mcfg["latent"],
    in_dim=mcfg.get("in_dim", 3),
    n_ctrl=ceil(config["training"]["max_frames"]//mcfg["frame_ctrl_ratio"])+mcfg["degree"],
    degree=mcfg["degree"],
    lam=mcfg["lam"],
).to(device)
model.load_state_dict(torch.load(os.path.join(load_dir, "SAE.pth"), map_location=device))
autocast = config["training"].get("autocast", False)

dataset = HeartcycleDataset(
    root='data/heartcycle',
    inputs=("echo",),            # only load echo
    include_time=True,
    echo_transpose_to_image=False,
    strict=False,                # skip files that don’t have echo
)


idx = random.randint(0, len(dataset) - 1)
sample = dataset[idx]
video = sample["x"]["echo"].to(device)
video = video.permute(1, 0, 2).float().div_(127.5).sub_(1).unsqueeze(0)  # [1, T, C, D]
timestamps = sample["t"]["echo"].to(device).unsqueeze(0)  # [1, T]

# Reconstruction
with torch.inference_mode():
    with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
        z = model.encode(video)
        z_spline = model.spline_fit_and_eval(z, timestamps, timestamps)
z = (z - z.mean(dim=1, keepdim=True)).squeeze(0).cpu().numpy()  # [T, D]
z_spline = (z_spline - z_spline.mean(dim=1, keepdim=True)).squeeze(0).cpu().numpy()  # [T, D]
timestamps = timestamps.squeeze(0).cpu().numpy()  # [T]

# Participation ratio of z
eigenvalues = np.var(z_spline, axis=0)
participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
print(f"Participation Ratio of z_spline: {participation_ratio:.4f}")

z = detrend(z, axis=0, type='linear')
z_spline = detrend(z_spline, axis=0, type='linear')


phase, evals, evecs, info = laplacian_phase(z_spline)

plot_phase_major_axis(z_spline, timestamps, phase, out_dir, idx)

# PCA for plotting
pca = PCA(n_components=3)
# z_spline = highpass_filter(z_spline, fs=fps, cutoff=0.5, order=4, axis=0)
# z_spline = preprocess_to_tangent_space(z_spline, pca=True)
z_spline_3d = pca.fit_transform(z_spline)

# plot_phase_and_z(z_spline_3d, phase, out_dir, idx, dim="2d", gt_ed=gt_ed, frames_idx=frames_idx)
plot_phase_and_z(z_spline_3d, phase, out_dir, idx, dim="3d")
plot_phase_and_time(phase, timestamps, out_dir, idx, differentiate=0)
# plot_phase_and_time(phase, timestamps, out_dir, idx, differentiate=1)
plot_znorm_and_time(z_spline_3d, timestamps, out_dir, idx)