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
from scipy.signal import find_peaks, detrend, savgol_filter
from sklearn.decomposition import PCA
from utils.find_extrema import compute_main_orientation_and_extrema
from utils.topology import cohomology_circular_coords, laplacian_phase, find_phase_major_axis, project_to_principal_plane


def EDES_via_LMP(z, z_spline, timestamps, fps, gt_ed, gt_es):
    group_ed, group_es, _, _, _, _ = compute_main_orientation_and_extrema(z_spline, fps)
    group = np.concatenate([group_ed, group_es])
    ed_err = np.min(np.abs(group - gt_ed))
    es_err = np.min(np.abs(group - gt_es))
    return ed_err, es_err


def EDES_via_Norm(z, z_spline, timestamps, fps, gt_ed, gt_es):
    pca = PCA(n_components=0.99)
    z = pca.fit_transform(z)
    z = np.linalg.norm(z, axis=-1)
    z = detrend(z, axis=0, type='linear')
    group = find_peaks(z, distance=5, prominence=0.2*(np.max(z)-np.min(z)))[0]
    ed_err = np.min(np.abs(group - gt_ed))
    es_err = np.min(np.abs(group - gt_es))
    return ed_err, es_err


def EDES_via_Phase(z, z_spline, timestamps, fps, gt_ed, gt_es):
    # phase = laplacian_phase(z_spline)[0]
    phase = cohomology_circular_coords(z_spline)[0]
    z_proj = z_spline @ find_phase_major_axis(z_spline, phase)
    z_proj = detrend(z_proj, type='linear')
    group1 = find_peaks(z_proj, prominence=0.2*(np.max(z_proj)-np.min(z_proj)), distance=5)[0]
    group2 = find_peaks(-z_proj, prominence=0.2*(np.max(z_proj)-np.min(z_proj)), distance=5)[0]
    group = np.concatenate([group1, group2])
    print(f"Length of video: {len(z_proj)} frames")
    print(f"Detected peaks at frames: {group}")
    print(f"Ground truth ES at frame {gt_es}, ED at frame {gt_ed}")
    ed_err = np.min(np.abs(group - gt_ed))
    es_err = np.min(np.abs(group - gt_es))
    return ed_err, es_err

def eval_split(dl, split_name: str, autocast: bool, detector_fn, max_workers=None):
    """
    Non-threaded version of eval_split.
    GPU inference + detector run sequentially in main process.
    """

    ed_mae_list, es_mae_list, fps_all = [], [], []

    with torch.inference_mode():
        for videos, timestamps, frames_idx, fps in tqdm(dl, desc=split_name):

            gt_es, gt_ed = frames_idx[0]
            fps = float(fps[0])
            gt_es = int(gt_es.item())
            gt_ed = int(gt_ed.item())

            videos = videos.to(device, non_blocking=True)
            timestamps = timestamps.to(device, non_blocking=True)
            # t0 = timestamps_cuda.min(); t1 = timestamps_cuda.max(); T = timestamps_cuda.shape[1]
            # dense_t = torch.linspace(t0, t1, (T-1)*4+1, device=device).unsqueeze(0)

            with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                z = model.encode(videos)
            z_spline = model.spline_fit_and_eval(z, timestamps, timestamps)

            # Move to CPU numpy
            z_np = z.squeeze(0).cpu().numpy()
            z_spline_np = z_spline.squeeze(0).cpu().numpy()
            timestamps_np = timestamps.squeeze(0).cpu().numpy()
            z_np = detrend(z_np, axis=0, type='linear')
            z_spline_np = detrend(z_spline_np, axis=0, type='linear')

            # Run detector inline
            ed_err, es_err = detector_fn(z_np, z_spline_np, timestamps_np, fps, gt_ed, gt_es)
            ed_mae_list.append(ed_err)
            es_mae_list.append(es_err)
            fps_all.append(fps)

    # ---- Aggregation (unchanged) ----
    mean_ed_mae = float(np.mean(ed_mae_list)) if len(ed_mae_list) else float("nan")
    mean_es_mae = float(np.mean(es_mae_list)) if len(es_mae_list) else float("nan")

    ed_ms = [mae * (1000.0 / fps) for mae, fps in zip(ed_mae_list, fps_all)]
    es_ms = [mae * (1000.0 / fps) for mae, fps in zip(es_mae_list, fps_all)]
    mean_ed_ms = np.mean(ed_ms) if len(ed_ms) else float("nan")
    mean_es_ms = np.mean(es_ms) if len(es_ms) else float("nan")

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

    load_dir = "results/2026_02_19/16_20_SAE"

    print("Starting ED/ES evaluation...")

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

    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=EDES_collate,
        num_workers=24,
        pin_memory=True
    )

    # ---- Evaluation ----
    detectors = [EDES_via_LMP, EDES_via_Norm, EDES_via_Phase]
    results = []

    for detector in detectors:
        try:
            results.append({"lines": [f"\n--- {detector.__name__} ---", ""]})
            results.append(eval_split(test_dl, "Test", autocast=autocast, detector_fn=detector, max_workers=3))
        except Exception as e:
            print(f"Error evaluating {detector.__name__}: {e}")
            results.append({"lines": [f"\n--- {detector.__name__} ---", "Evaluation Failed"]})

    # ---- Save results ----
    with open(os.path.join(load_dir, "edes_detection.txt"), "w") as f:
        for r in results:
            f.write(r["lines"][0] + "\n")
            f.write(r["lines"][1] + "\n")

    print("ED/ES Evaluation Complete.")
