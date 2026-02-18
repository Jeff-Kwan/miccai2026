# =========================
# Topological loop analysis (cohomology-based circular coordinates) of the SAE latent space
# =========================
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch 
from models.SplineAutoEncoder import SplineAutoEncoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
import os
import json
from math import ceil
from sklearn.decomposition import PCA
from utils.topology import cohomology_circular_coords, plot_phase_and_z, plot_phase_and_time, \
        plot_znorm_and_time, plot_phase_major_axis, highpass_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_17/18_35_SAE"
out_dir = os.path.join(load_dir, "topology")
os.makedirs(out_dir, exist_ok=True)


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


idx = 1150 #random.randint(0, len(val_ds) - 1)
video = val_ds[idx]['video'].to(device).unsqueeze(0)
video = video * 2 - 1  # [0,1] → [-1,1]
timestamps = val_ds[idx]['timestamps'].unsqueeze(0).to(device)
frames_idx = val_ds[idx]['masks']['frame_indices']
gt_es, gt_ed = frames_idx
fps = val_ds[idx]["metadata"]["FPS"]

# Upsample via spline interpolation
t0 = timestamps.min(); t1 = timestamps.max(); T = video.shape[1]
dense_factor = 1
t_dense = torch.linspace(t0, t1, (T-1)*dense_factor+1, device=device).unsqueeze(0)


# Reconstruction
B, T, C, H, W = video.shape
with torch.inference_mode():
    with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
        z = model.encode(video)
        z_spline = model.spline_fit_and_eval(z, timestamps, t_dense)
z = (z - z.mean(dim=1, keepdim=True)).squeeze(0).cpu().numpy()  # [T, D]
z_spline = (z_spline - z_spline.mean(dim=1, keepdim=True)).squeeze(0).cpu().numpy()  # [T, D]
timestamps = timestamps.squeeze(0).cpu().numpy()  # [T]
t_dense = t_dense.squeeze(0).cpu().numpy()  # [T_dense]

phase, dgms = cohomology_circular_coords(
    z_spline, fps=fps,
    print_dgms_summary=True) 

# phase = phase[::dense_factor]  # Downsample back

plot_phase_major_axis(z_spline, t_dense, phase, out_dir, idx, frames_idx=frames_idx)

# PCA for plotting
pca = PCA(n_components=3)
z_spline = highpass_filter(z_spline, fs=fps*dense_factor, cutoff=0.5, order=4, axis=0)  # High-pass filter to remove drift
z_spline_3d = pca.fit_transform(z_spline)

# plot_phase_and_z(z_spline_3d, phase, out_dir, idx, dim="2d", gt_ed=gt_ed, frames_idx=frames_idx)
plot_phase_and_z(z_spline_3d, phase, out_dir, idx, dim="3d", gt_ed=gt_ed, frames_idx=frames_idx)
plot_phase_and_time(phase, t_dense, out_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx, differentiate=0)
# plot_phase_and_time(phase, t_dense, out_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx, differentiate=1)
plot_znorm_and_time(z_spline_3d, t_dense, out_dir, idx, frames_idx=frames_idx)