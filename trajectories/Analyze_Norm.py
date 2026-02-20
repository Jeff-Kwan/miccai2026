import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from Dataset import get_latents_dataset
from scipy.signal import detrend
from sklearn.decomposition import PCA

train_ds, val_ds, test_ds = get_latents_dataset()


norms_list = []
norms_diff = []
EF_list = []
ESV_list = []
EDV_list = []
for sample in tqdm(val_ds):
    z_spline = sample["z_spline"]  # [T, latent_dim]
    z_spline = detrend(z_spline, axis=0, type="linear")
    gt_ed, gt_es = sample["frame_indices"]
    EF_list.append(sample["EF"])
    ESV_list.append(sample["ESV"])
    EDV_list.append(sample["EDV"])
    norm = np.linalg.norm(z_spline[gt_es] - z_spline[gt_ed], axis=-1)
    # norms = detrend(norms, axis=0, type="linear")
    norms_diff.append(norm)

# norms_array = np.concatenate(norms_list, axis=0)
norms_diff_array = np.array(norms_diff)  # [N]
EF_array = np.array(EF_list)
ESV_array = np.array(ESV_list)
EDV_array = np.array(EDV_list)

# Try linear regression of norms_diff vs EF, ESV, EDV
from sklearn.linear_model import LinearRegression
X = norms_diff_array.reshape(-1, 1)
y_EF = EF_array
y_ESV = ESV_array
y_EDV = EDV_array
reg_EF = LinearRegression().fit(X, y_EF)
reg_ESV = LinearRegression().fit(X, y_ESV)
reg_EDV = LinearRegression().fit(X, y_EDV)

print("Linear regression of Norms Diff vs EF:")
print(f"Coefficient: {reg_EF.coef_[0]:.4f}, Intercept: {reg_EF.intercept_:.4f}, R^2: {reg_EF.score(X, y_EF):.4f}")
print("Linear regression of Norms Diff vs ESV:")
print(f"Coefficient: {reg_ESV.coef_[0]:.4f}, Intercept: {reg_ESV.intercept_:.4f}, R^2: {reg_ESV.score(X, y_ESV):.4f}")
print("Linear regression of Norms Diff vs EDV:")
print(f"Coefficient: {reg_EDV.coef_[0]:.4f}, Intercept: {reg_EDV.intercept_:.4f}, R^2: {reg_EDV.score(X, y_EDV):.4f}")