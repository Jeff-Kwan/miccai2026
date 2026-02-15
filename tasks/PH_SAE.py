# =========================
# Persistent homology (PH)
# =========================
# pip install ripser persim scikit-learn
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch 
from models.SplineAutoEncoder import SplineAutoEncoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import json
from math import ceil
from ripser import ripser
from persim import plot_diagrams
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_15/07_33_SAE"
ph_dir = os.path.join(load_dir, "persistent_homology")
os.makedirs(ph_dir, exist_ok=True)


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

train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=True)


idx = random.randint(0, len(val_ds) - 1)
video = val_ds[idx]['video'].to(device).unsqueeze(0)
video = video * 2 - 1  # [0,1] → [-1,1]
timestamps = val_ds[idx]['timestamps'].unsqueeze(0).to(device)
frames_idx = val_ds[idx]['masks']['frame_indices']
t0 = timestamps.min(); t1 = timestamps.max()    # 100 fps
t_dense = torch.linspace(t0, t1, steps=int(100*(t1-t0)), device=device).unsqueeze(0)    


# Reconstruction
B, T, C, H, W = video.shape
with torch.inference_mode():
    with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
        z = model.encode(video)
        z = model.spline_fit_and_eval(z, timestamps, t_dense)
        reconstruction = model.decode(z, H, W)

# ---- Choose what you want to analyze as a point cloud ----
Z_np = z[0].detach().float().cpu().numpy()            # [T, D]

# Center across time (remove static component)
Z_np = Z_np - Z_np.mean(axis=0, keepdims=True)

# PCA and keep top k components
pca = PCA(n_components=0.95, svd_solver='full')
Z_np = pca.fit_transform(Z_np)  # [T, k]
print(f"Original latent dim: {z.shape[2]}, after PCA (95% variance): {Z_np.shape[1]}")

# ---- Run Ripser on the point cloud ----
# maxdim=2 gives H0/H1/H2.
res = ripser(Z_np, maxdim=2)
dgms = res["dgms"]  # Raw diagrams

# ---- Plot persistence diagrams ----
fig = plt.figure(figsize=(6, 6))
plot_diagrams(dgms, show=False)
plt.title("Persistence diagrams (Ripser)")
plt.tight_layout()
plt.savefig(os.path.join(ph_dir, "persistence_diagrams.png"))
plt.close(fig)

# ---- Quick quantitative summary (largest H1 persistence) ----
def top_persistences(diagram: np.ndarray, k: int = 5):
    # diagram rows: [birth, death] ; death can be inf (H0)
    if diagram.size == 0:
        return []
    b = diagram[:, 0]
    d = diagram[:, 1]
    finite = np.isfinite(d)
    pers = (d - b)
    pers = pers[finite]
    if pers.size == 0:
        return []
    pers_sorted = np.sort(pers)[::-1]
    return pers_sorted[:k].tolist()

H0 = dgms[0]
H1 = dgms[1]
H2 = dgms[2] if len(dgms) > 2 else np.zeros((0, 2))

summary = {
    "H0_count": int(H0.shape[0]),
    "H1_count": int(H1.shape[0]),
    "H2_count": int(H2.shape[0]),
    "H1_top_persistences": top_persistences(H1, k=10),
    "H2_top_persistences": top_persistences(H2, k=10),
}

with open(os.path.join(ph_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("PH summary:", summary)
