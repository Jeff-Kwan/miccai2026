import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
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
from scipy.signal import find_peaks
from utils.find_extrema import compute_main_orientation_and_extrema
from utils.topology import cohomology_circular_coords, find_phase_major_axis, plot_phase_major_axis


def EDES_via_LMP(z, z_spline, timestamps, fps):
    group_ed, group_es, _, _, _, _ = compute_main_orientation_and_extrema(z_spline, fps)
    return np.concatenate([group_ed, group_es])


def EDES_via_Norm(z, z_spline, timestamps, fps):
    z = np.linalg.norm(z, axis=-1)
    return find_peaks(z, prominence=0.5*np.std(z))[0]


def EDES_via_Phase(z, z_spline, timestamps, fps):
    phase, dgms = cohomology_circular_coords(
        z_spline, fps=fps, 
        savgol=False, highpass=True, pca=True,
        print_dgms_summary=False)
    major_axis = find_phase_major_axis(z, phase)
    z_proj = z_spline @ major_axis
    group1 = find_peaks(z_proj, prominence=0.5*np.std(z_proj))[0]
    group2 = find_peaks(-z_proj, prominence=0.5*np.std(z_proj))[0]
    return np.concatenate([group1, group2])


def eval_split(dl, split_name: str, autocast: bool, detector_fn):
    ed_mae_list, es_mae_list, fps_all = [], [], []
    with torch.inference_mode():
        for videos, timestamps, frames_idx, fps in tqdm(dl, desc=split_name):
            gt_es, gt_ed = frames_idx[0]
            fps = float(fps[0])
            gt_es = int(gt_es.item())
            gt_ed = int(gt_ed.item())
            videos = videos.to(device, non_blocking=True)
            timestamps = timestamps.to(device, non_blocking=True)

            with torch.inference_mode():
                with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                    z = model.encode(videos)
                    z_spline = model.spline_fit_and_eval(z, timestamps, timestamps)
            z = (z - z.mean(dim=1, keepdim=True)).squeeze(0).cpu().numpy()  # [T, D]
            z_spline = (z_spline - z_spline.mean(dim=1, keepdim=True)).squeeze(0).cpu().numpy()  # [T, D]

            # HOW do I know which is which??!
            group = detector_fn(z, z_spline, timestamps, fps)

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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # parser = argparse.ArgumentParser(description="Evaluate ED/ES detection using Spline AutoEncoder features.")
    # parser.add_argument("--load_dir", type=str, required=True, help="Directory to load the trained Spline AutoEncoder model from.")
    # args = parser.parse_args()
    # load_dir = args.load_dir
    load_dir = "results/2026_02_17/17_57_SAE"

    print("Starting ED/ES evaluation...")
    # ---- Model ----
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
    model = model.to(device).eval()

    # ---- Dataset ----
    train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=True)

    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
        collate_fn=EDES_collate, num_workers=24, pin_memory=True)

    # Evaluation
    detectors = [EDES_via_LMP, EDES_via_Norm, EDES_via_Phase]
    results = []
    for detector in detectors:
        try:
            results.append({"lines": [f"\n--- {detector.__name__} ---", ""]})
            results.append(eval_split(test_dl, "Test", autocast=autocast, detector_fn=detector))
        except Exception as e:
            print(f"Error evaluating {detector.__name__}: {e}")
            results.append({"lines": [f"\n--- {detector.__name__} ---", "Evaluation Failed"]})

    # ---- Save to <load_dir>/edes_detection.txt ----
    with open(os.path.join(load_dir, "edes_detection.txt"), "w") as f:
        for r in results:
            f.write(r["lines"][0] + "\n")
            f.write(r["lines"][1] + "\n")

    print("ED/ES Evaluation Complete.")