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
import umap
from utils.find_extrema import compute_main_orientation_and_extrema
from utils.topology import cohomology_circular_coords, laplacian_phase, find_phase_major_axis,\
project_to_phase_plane, von_mises_kernel_smoother

def find_peaks_sentinel(input_array, p, d):
    prominence = p * (np.max(input_array) - np.min(input_array))
    # Sentinel padding to allow edge peaks/valleys to be detected
    peak_input   = np.concatenate(([np.min(input_array)], input_array, [np.min(input_array)]))
    valley_input = np.concatenate(([np.max(input_array)], input_array, [np.max(input_array)]))

    peaks = find_peaks(peak_input, prominence=prominence, distance=d)[0]      # maxima
    valleys = find_peaks(-valley_input, prominence=prominence, distance=d)[0]   # minima

    # Remove sentinel offset and filter out invalid indices
    peaks = peaks - 1
    valleys = valleys - 1
    N = len(input_array)
    peaks = peaks[(peaks >= 0) & (peaks < N)]
    valleys = valleys[(valleys >= 0) & (valleys < N)]

    return peaks, valleys

def EDES_via_LMP(z, z_spline, timestamps, fps, gt_ed, gt_es):
    group_ed, group_es, _, _, _, _ = compute_main_orientation_and_extrema(z_spline, fps)
    group = np.concatenate([group_ed, group_es])
    ed_err = np.min(np.abs(group - gt_ed))
    es_err = np.min(np.abs(group - gt_es))
    return ed_err, es_err


def EDES_via_Norm(z, z_spline, timestamps, fps, gt_ed, gt_es):
    phase = laplacian_phase(z_spline)[0]
    grid, mu = von_mises_kernel_smoother(z_spline, phase, n_grid=512, kappa=1)
    mu = np.linalg.norm(mu, axis=1)
    peaks, _ = find_peaks_sentinel(mu, p=0.2, d=5)
    pred_phases = grid[peaks]
    delta1 = np.abs(phase - pred_phases[0])
    delta2 = np.abs(phase - pred_phases[1])
    _, valleys1 = find_peaks_sentinel(delta1, p=0.2, d=5)
    _, valleys2 = find_peaks_sentinel(delta2, p=0.2, d=5)
    group = np.concatenate([valleys1, valleys2])
    ed_err = np.min(np.abs(group - gt_ed))
    es_err = np.min(np.abs(group - gt_es))
    return ed_err, es_err


def EDES_via_Phase(z, z_spline, timestamps, fps, gt_ed, gt_es, edge_events=True):
    # phase = cohomology_circular_coords(z_spline, print_dgms_summary=False)[0]
    phase = laplacian_phase(z_spline)[0]
    z_proj = z @ find_phase_major_axis(z, phase)
    z_proj = detrend(z_proj, type='linear')
    peaks, valleys = find_peaks_sentinel(z_proj, p=0.2, d=5)
    group = np.concatenate([peaks, valleys])
    ed_err = np.min(np.abs(group - gt_ed))
    es_err = np.min(np.abs(group - gt_es))
    return ed_err, es_err


def EDES_via_UMAP(z, z_spline, timestamps, fps, gt_ed, gt_es):
    umap_model = umap.UMAP(
    n_components=1,
    n_neighbors=15,     # try 5–50 (local ↔ global)
    min_dist=0.2,       # smaller = tighter clusters
    metric="euclidean")
    z_umap = umap_model.fit_transform(z_spline).squeeze()
    z_umap = detrend(z_umap, axis=0, type='linear')
    prom = 0.2 * (np.max(z_umap) - np.min(z_umap))
    # Sentinel padding to allow edge peaks/valleys to be detected
    peak_input   = np.concatenate(([np.min(z_umap)], z_umap, [np.min(z_umap)]))
    valley_input = np.concatenate(([np.max(z_umap)], z_umap, [np.max(z_umap)]))
    group1 = find_peaks(peak_input, prominence=prom, distance=5)[0]      # maxima
    group2 = find_peaks(-valley_input, prominence=prom, distance=5)[0]   # minima
    group1 = group1 - 1
    group2 = group2 - 1
    N = len(z_umap)
    group1 = group1[(group1 >= 0) & (group1 < N)]
    group2 = group2[(group2 >= 0) & (group2 < N)]
    group = np.concatenate([group1, group2])
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

    load_dir = "results/2026_02_21/17_26_SAE"

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
    detectors = [EDES_via_Norm]
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
