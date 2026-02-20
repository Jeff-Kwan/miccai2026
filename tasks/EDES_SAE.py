import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from datahandling.collate import EDES_collate
from models.SplineAutoEncoder import SplineAutoEncoder
import os
import numpy as np
from tqdm import tqdm
import json
from math import ceil
from utils.find_extrema import compute_main_orientation_and_extrema, savgol_filter, highpass_filter, detect_baseline_wander
from utils.filters import highpass, SavGolFilterTime, bandpass, gaussian_smooth
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_17/12_03_SAE"

# ---- Model ----
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
model = model.to(device).eval()

# ---- Dataset ----
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=True)

train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
    collate_fn=EDES_collate, num_workers=24, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True,
    collate_fn=EDES_collate, num_workers=24, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
    collate_fn=EDES_collate, num_workers=24, pin_memory=True)

def eval_split(dl, split_name: str, use_amp: bool = True):
    ed_mae_list, es_mae_list, fps_all = [], [], []
    savgol = SavGolFilterTime(window_length=11, polyorder=3)
    with torch.inference_mode():
        for videos, timestamps, frames_idx, fps in tqdm(dl, desc=split_name):
            # [0] because we go through one by one (variable length video)
            gt_es, gt_ed = frames_idx[0]
            fps = float(fps[0])
            gt_es = int(gt_es.item())
            gt_ed = int(gt_ed.item())
            videos = videos.to(device, non_blocking=True)
            timestamps = timestamps.to(device, non_blocking=True)
            
            with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
                z = model.encode(videos)
            # z = model.spline_fit_and_eval(z, timestamps, timestamps)
            z_motion = (z - z.mean(dim=1, keepdim=True)) # Remove static component
            # z_motion = savgol(z_motion.squeeze(0))
            # z_motion = highpass(z_motion, fs=fps, cutoff=0.5, order=2)
            z_motion = z_motion.float().squeeze().cpu().numpy()

            # HOW do I know which is which??!
            # group_ed, group_es, _, _, _, _ = compute_main_orientation_and_extrema(z_motion, fps)
            # group = np.concatenate([group_ed, group_es])

            z_motion = np.linalg.norm(z_motion, axis=-1)
            # z_motion = savgol_filter(z_motion, 7, 3, axis=0)
            group = find_peaks(z_motion, prominence=0.5*np.std(z_motion))[0]

            # Mean Absolute Error
            ed_err = np.min(np.abs(group - gt_ed))
            es_err = np.min(np.abs(group - gt_es))

            ed_mae_list.append(ed_err)
            es_mae_list.append(es_err)
            fps_all.append(fps)


    mean_ed_mae = float(np.mean(ed_mae_list)) if len(ed_mae_list) else float("nan")
    mean_es_mae = float(np.mean(es_mae_list)) if len(es_mae_list) else float("nan")

    # Convert per-frame mae to ms using sample-FPS
    ed_ms = [mae * (1000.0 / fps) for mae, fps in zip(ed_mae_list, fps_all)]
    es_ms = [mae * (1000.0 / fps) for mae, fps in zip(es_mae_list, fps_all)]
    mean_ed_ms = np.mean(ed_ms) if len(ed_ms) else float("nan")
    mean_es_ms = np.mean(es_ms) if len(es_ms) else float("nan")

    # "Same format as now"
    out_lines = [
        f"{split_name} ED MAE: {mean_ed_mae:.2f} frames, {mean_ed_ms:.2f} ms",
        f"{split_name} ES MAE: {mean_es_mae:.2f} frames, {mean_es_ms:.2f} ms\n",
    ]

    print(out_lines[0])
    print(out_lines[1])

    return {
        "split": split_name,
        "ed_frames": mean_ed_mae,
        "ed_ms": mean_ed_ms,
        "es_frames": mean_es_mae,
        "es_ms": mean_es_ms,
        "lines": out_lines
    }

results = []
# results.append(eval_split(train_dl, "Train", use_amp=autocast))
# results.append(eval_split(val_dl, "Val", use_amp=autocast)) 
results.append(eval_split(test_dl, "Test", use_amp=autocast))

# ---- Save to <load_dir>/edes_detection.txt ----
out_path = os.path.join(load_dir, "edes_detection.txt")
with open(out_path, "w") as f:
    for r in results:
        f.write(r["lines"][0] + "\n")
        f.write(r["lines"][1] + "\n")

print(f"\nSaved results to: {out_path}")