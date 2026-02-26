import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
from scipy.signal import detrend

from Dataset import get_latents_dataset

from sklearn.neighbors import NearestNeighbors

# Load data
train_ds, val_ds, test_ds = get_latents_dataset()

def compute_pr(z):
    # z: [T, latent_dim]
    cov = np.cov(z, rowvar=False)  # [latent_dim, latent_dim]
    eigenvalues = np.linalg.eigvalsh(cov)  # [latent_dim]
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr

def compute_local_pr(z, k=20, temporal_exclusion=5):
    T, D = z.shape
    nbrs = NearestNeighbors(n_neighbors=T)
    nbrs.fit(z)
    _, indices = nbrs.kneighbors(z)   # (T, T)
    i = np.arange(T)[:, None]         # (T, 1)
    valid = np.abs(indices - i) > temporal_exclusion
    filtered = np.where(valid, indices, -1)
    order = np.argsort(filtered == -1, axis=1)
    filtered = np.take_along_axis(filtered, order, axis=1)
    idx = filtered[:, :k]             # (T, k))
    idx_safe = np.where(idx == -1, 0, idx)
    z_local = z[idx_safe]             # (T, k, D)
    mask = (idx != -1)[..., None]
    counts = mask.sum(axis=1, keepdims=True).clip(min=1)
    mean = (z_local * mask).sum(axis=1, keepdims=True) / counts
    centered = (z_local - mean) * mask
    denom = np.maximum(counts.squeeze() - 1, 1)
    cov = np.einsum("tkd,tke->tde", centered, centered)
    cov = cov / denom[:, None, None]
    eig = np.linalg.eigvalsh(cov)     # (T, D)
    s1 = eig.sum(axis=1)
    s2 = (eig ** 2).sum(axis=1)
    pr = np.where(s2 < 1e-12, 0.0, (s1 ** 2) / s2)
    return pr.mean()

participation_ratios = []
local_prs = []
z_list = []

for sample in tqdm(test_ds):
    z = sample["z"]          # [T, latent_dim]
    participation_ratios.append(compute_pr(z))
    local_prs.append(compute_local_pr(z))
    z_list.append(z)

z_global = np.concatenate(z_list)
print(f"Sample-Averaged Participation Ratios: {np.mean(participation_ratios)} ± {np.std(participation_ratios)}")
print(f"Sample-Averaged Local Participation Ratios: {np.mean(local_prs)} ± {np.std(local_prs)}")
print(f"Global Participation Ratio: {compute_pr(z_global)}")