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
from utils.topology import cohomology_circular_coords, plot_phase_and_z, plot_phase_and_time, plot_znorm_and_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_17/11_58_SAE"
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
fps = val_ds[idx]["metadata"]["FPS"].item()
# t0 = timestamps.min(); t1 = timestamps.max()    # 100 fps
# t_dense = torch.linspace(t0, t1, steps=int(200*(t1-t0)), device=device).unsqueeze(0)    


# Reconstruction
B, T, C, H, W = video.shape
with torch.inference_mode():
    with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
        z = model.encode(video)
        z_spline = model.spline_fit_and_eval(z, timestamps, timestamps)
z = z.squeeze(0).cpu().numpy()  # [T, D]
z_spline = z_spline.squeeze(0).cpu().numpy()  # [T, D]

phase, dgms = cohomology_circular_coords(
    z_spline, fps=fps, 
    savgol=False, highpass=True, pca=True, 
    print_dgms_summary=True) 

plot_phase_and_z(z_spline, phase, ph_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx)
plot_phase_and_time(phase, timestamps.squeeze(0).cpu().numpy(), ph_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx, differentiate=0)
plot_phase_and_time(phase, timestamps.squeeze(0).cpu().numpy(), ph_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx, differentiate=1)
plot_znorm_and_time(z_spline, timestamps.squeeze(0).cpu().numpy(), ph_dir, idx, gt_ed=gt_ed, frames_idx=frames_idx)