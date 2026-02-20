import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from Dataset import get_latents_dataset
from scipy.signal import detrend, find_peaks
from sklearn.decomposition import PCA

from utils.topology import find_phase_major_axis

train_ds, val_ds, test_ds = get_latents_dataset()


ED_err_list = []
ES_err_list = []
large_error_idx = []
for i, sample in tqdm(enumerate(val_ds)):
    z_spline = sample["z_spline"]  # [T, latent_dim]
    z_spline = detrend(z_spline, axis=0, type="linear")
    phase = sample["phase"]  # [T]
    gt_ed, gt_es = sample["frame_indices"]

    z_proj = z_spline @ find_phase_major_axis(z_spline, phase)
    z_proj = detrend(z_proj, type='linear')
    group1 = find_peaks(z_proj, prominence=0.2*(np.max(z_proj)-np.min(z_proj)), distance=5)[0]
    group2 = find_peaks(-z_proj, prominence=0.2*(np.max(z_proj)-np.min(z_proj)), distance=5)[0]
    group = np.concatenate([group1, group2])
    ed_err = np.min(np.abs(group - gt_ed))
    es_err = np.min(np.abs(group - gt_es))
    ED_err_list.append(ed_err)
    ES_err_list.append(es_err)

    if ed_err > 10 or es_err > 10:
        large_error_idx.append(i)

ED_err_array = np.array(ED_err_list)
ES_err_array = np.array(ES_err_list)

print("ED error stats:")
print(f"Min: {ED_err_array.min():.4f}")
print(f"Max: {ED_err_array.max():.4f}")
print(f"Median: {np.median(ED_err_array):.4f}")
print(f"Mean: {ED_err_array.mean():.4f}")
print(f"Std: {ED_err_array.std():.4f}")

print("ES error stats:")
print(f"Min: {ES_err_array.min():.4f}")
print(f"Max: {ES_err_array.max():.4f}")
print(f"Median: {np.median(ES_err_array):.4f}")
print(f"Mean: {ES_err_array.mean():.4f}")
print(f"Std: {ES_err_array.std():.4f}")

print(f"Large error indices (ED or ES error > 5 frames): \n{large_error_idx}")