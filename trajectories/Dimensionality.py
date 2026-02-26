import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
from scipy.signal import detrend

from Dataset import get_latents_dataset

from sklearn.linear_model import Ridge

# Load data
train_ds, val_ds, test_ds = get_latents_dataset()

def compute_pr(z):
    # z: [T, latent_dim]
    cov = np.cov(z, rowvar=False)  # [latent_dim, latent_dim]
    eigenvalues = np.linalg.eigvalsh(cov)  # [latent_dim]
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr

participation_ratios = []
z_list = []

for sample in tqdm(test_ds):
    z = sample["z"]          # [T, latent_dim]
    participation_ratios.append(compute_pr(z))
    z_list.append(z)

z_global = np.concatenate(z_list)
print(f"Sample-Averaged Participation Ratios: {np.mean(participation_ratios)} ± {np.std(participation_ratios)}")
print(f"Global Participation Ratio: {compute_pr(z_global)}")