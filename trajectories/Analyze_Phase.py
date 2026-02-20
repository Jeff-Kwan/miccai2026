import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from Dataset import get_latents_dataset
from scipy.signal import detrend
from sklearn.decomposition import PCA

train_ds, val_ds, test_ds = get_latents_dataset()


phase_diff_list = []
for sample in tqdm(val_ds):
    phase = sample["phase"]  # [T]
    gt_ed, gt_es = sample["frame_indices"]
    phase_diff_list.append(((phase[gt_es] - phase[gt_ed])) / (np.pi))

phase_diff_array = np.abs(np.stack(phase_diff_list))  # [N]

# Min, max, median, mean, std of phase diff
print("ABS Phase diff stats:")
print(f"Min: {phase_diff_array.min():.4f}")
print(f"Max: {phase_diff_array.max():.4f}")
print(f"Median: {np.median(phase_diff_array):.4f}")
print(f"Mean: {phase_diff_array.mean():.4f}")
print(f"Std: {phase_diff_array.std():.4f}")

# Histogram of phase differences
plt.figure(figsize=(8, 6))
plt.hist(phase_diff_array, bins=20, density=True, alpha=0.7)
plt.title("Histogram of ABS Phase Differences (normalized by pi)")
plt.xlabel("ABS Phase Difference / pi")
plt.ylabel("Density")
plt.grid()
plt.savefig("phase_diff_histogram.png")