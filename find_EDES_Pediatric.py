import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from datahandling.EchoPediaDatasetShard import get_echopedia_shard_dataset
from datahandling.collate import EDES_collate
from models.SplineAutoEncoder import SplineAutoEncoder
import os
import numpy as np
from tqdm import tqdm
import json
from tasks.Compute_EDES import EDES_via_Phase, EDES_via_LMP, EDES_via_Norm
from scipy.signal import detrend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_24/14_57_SAE"

# ---- Model ----
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
model = model.to(device).eval()

# ---- Dataset ----
a4c_ds = get_echopedia_shard_dataset(views="A4C")
psax_ds = get_echopedia_shard_dataset(views="PSAX")

a4c_dl = DataLoader(
    a4c_ds,
    batch_size=1,
    shuffle=False,
    collate_fn=EDES_collate,
    num_workers=24,
    pin_memory=True)

psax_dl = DataLoader(
    psax_ds,
    batch_size=1,
    shuffle=False,
    collate_fn=EDES_collate,
    num_workers=24,
    pin_memory=True)

### Run
def run_dl(dl, view_name):
    print(f"Running ED/ES detection on {view_name} view...")
    problems = {}; errors = []; ms_errors = []; assignments = []; min_err_list = []
    with torch.inference_mode():
        for i, batch in tqdm(enumerate(dl)):
            videos, timestamps, fps, ed, es = batch
            fps = float(fps[0])
            gt_es = int(ed[0])
            gt_ed = int(es[0])

            videos = videos.to(device, non_blocking=True)

            with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                z = model.encode(videos)
            z = z.squeeze(0).cpu().numpy()
            z = detrend(z, axis=0, type='linear')

            try:
                ed_err, es_err, assign, min_err = EDES_via_Phase(z, gt_ed, gt_es)
                # if ed_err == 0 and es_err == 0:
                #     print(f"Sample {i} has no errors: ED_err={ed_err}, ES_err={es_err}")
                errors.append([i, ed_err, es_err])
                assignments.append(assign)
                min_err_list += min_err
                if fps is not None:
                    ed_ms_err = ed_err * (1000.0 / fps)
                    es_ms_err = es_err * (1000.0 / fps)
                    ms_errors.append([i, ed_ms_err, es_ms_err])
                # print(f"Sample index {i}: ED error = {ed_err:.3f} frames, ES error = {es_err:.3f} frames")
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

    print(f"Localization error = {np.mean(min_err_list):.3f} frames, STD: {np.std(min_err_list):.3f} frames")
    print(f"Correct assignments: {sum(assignments) / len(assignments) * 100:.3f}%")
    print(f"ED Error MAE: {np.mean([e[1] for e in errors]):.3f} frames, STD: {np.std([e[1] for e in errors]):.3f} frames")
    print(f"ED Error Time: {np.mean([e[1] for e in ms_errors]):.3f} ms, STD: {np.std([e[1] for e in ms_errors]):.3f} ms")
    print(f"ES Error MAE: {np.mean([e[2] for e in errors]):.3f} frames, STD: {np.std([e[2] for e in errors]):.3f} frames")
    print(f"ES Error Time: {np.mean([e[2] for e in ms_errors]):.3f} ms, STD: {np.std([e[2] for e in ms_errors]):.3f} ms")
    # print indices & errors of first 3 largest errors
    # errors = np.array(errors)
    # largest_errors = errors[np.argsort(errors[:, 1])[::-1][:3]]
    # print("\nLargest ED errors:")
    # for idx, ed_err, es_err in largest_errors:
    #     print(f"Sample index {idx}: ED error = {ed_err:.3f} frames, ES error = {es_err:.3f} frames")
    # largest_errors = errors[np.argsort(errors[:, 2])[::-1][:3]]
    # print("\nLargest ES errors:")
    # for idx, ed_err, es_err in largest_errors:
    #     print(f"Sample index {idx}: ED error = {ed_err:.3f} frames, ES error = {es_err:.3f} frames")

run_dl(a4c_dl, "A4C")
run_dl(psax_dl, "PSAX")