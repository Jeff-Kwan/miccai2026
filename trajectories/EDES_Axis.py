import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from Dataset import get_latents_dataset


train_ds, val_ds, test_ds = get_latents_dataset()

z_ED_list = []
z_ES_list = []
for i, sample in tqdm(enumerate(train_ds)):
    z = sample["z"]  # [T, latent_dim]
    gt_ed = sample["ED"]
    gt_es = sample["ES"]
    fps = sample["metadata"]["FPS"]

    z_ED_list.append(z[gt_ed])
    z_ES_list.append(z[gt_es])

z_ED = np.array(z_ED_list)
z_ES = np.array(z_ES_list)

diff = z_ES - z_ED                         # [N, D]
d = diff.mean(axis=0)                      # [D]
d = d / np.linalg.norm(d)

proj_ED = z_ED @ d
proj_ES = z_ES @ d
margins = proj_ES - proj_ED                # [N]

acc = (margins > 0).mean()
print(f"Ordering accuracy (ED < ES along d): {acc:.3f}")
print(f"Mean margin: {margins.mean():.4f}, median: {np.median(margins):.4f}")
print(f"Frac margins > 0: {(margins>0).mean():.3f}")

diff_norm = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12)
cosines = diff_norm @ d

print("Mean cosine:", cosines.mean())
print("Median cosine:", np.median(cosines))

# Save d
np.save("tasks/EDES_axis.npy", d)