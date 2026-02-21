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
from tasks.Compute_EDES import EDES_via_Phase, EDES_via_LMP, EDES_via_Norm
from scipy.signal import detrend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_20/16_42_SAE"

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

dl = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    collate_fn=EDES_collate,
    num_workers=24,
    pin_memory=True
)

### Run
problems = {}; errors = []
with torch.inference_mode():
    for i, batch in tqdm(enumerate(dl)):
        # if i < 80:
        #     continue
        videos, timestamps, frames_idx, fps = batch
        gt_es, gt_ed = frames_idx[0]
        fps = float(fps[0])
        gt_es = int(gt_es.item())
        gt_ed = int(gt_ed.item())

        videos = videos.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)

        with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
            z = model.encode(videos)
            z_spline = model.spline_fit_and_eval(z, timestamps, timestamps)
        z = z.squeeze(0).cpu().numpy()
        z_spline = z_spline.squeeze(0).cpu().numpy()
        timestamps = timestamps.squeeze(0).cpu().numpy()
        z = detrend(z, axis=0, type='linear')
        z_spline = detrend(z_spline, axis=0, type='linear')
        try:
            ed_err, es_err = EDES_via_Phase(z, z_spline, timestamps, fps, gt_ed, gt_es)
            errors.append([i, ed_err, es_err])
            # print(f"Sample index {i}: ED error = {ed_err:.2f} frames, ES error = {es_err:.2f} frames")
            # exit()
        except Exception as e:
            print("!!!!!")
            print(f"Sample index {i}: Video Length {videos.size(1)} : {e}")
            print("!!!!!")
            key = (type(e).__name__, str(e))
            if key not in problems:
                problems[key] = [i]
            else:
                problems[key].append(i)

print("\nSummary of problems encountered:")
for error, indices in problems.items():
    print(f"Error: {error} - Occurred in samples: {indices}")

print(f"ED Error MAE: {np.mean([e[1] for e in errors]):.2f} frames, STD: {np.std([e[1] for e in errors]):.2f} frames")
print(f"ES Error MAE: {np.mean([e[2] for e in errors]):.2f} frames, STD: {np.std([e[2] for e in errors]):.2f} frames")
# print indices & errors of first 3 largest errors
errors = np.array(errors)
largest_errors = errors[np.argsort(errors[:, 1])[::-1][:3]]
print("\nLargest ED errors:")
for idx, ed_err, es_err in largest_errors:
    print(f"Sample index {idx}: ED error = {ed_err:.2f} frames, ES error = {es_err:.2f} frames")
largest_errors = errors[np.argsort(errors[:, 2])[::-1][:3]]
print("\nLargest ES errors:")
for idx, ed_err, es_err in largest_errors:
    print(f"Sample index {idx}: ED error = {ed_err:.2f} frames, ES error = {es_err:.2f} frames")