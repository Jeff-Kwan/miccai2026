import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
from scipy.signal import detrend, savgol_filter

from Dataset import get_latents_dataset
from utils.topology import project_to_major_axis, von_mises_kernel_smoother
from tasks.Compute_EDES import find_peaks_sentinel

from sklearn.linear_model import Ridge

EDES_axis = np.load("tasks/EDES_axis.npy")

def build_X_y(ds, desc="dataset"):
    features_list = []
    EF_list = []

    for sample in tqdm(ds, desc=f"Featurizing {desc}"):
        z = sample["z"]          # [T, latent_dim]
        phase = sample["phase"]  # [T]

        # Preprocess
        z = detrend(z, axis=0, type="linear")

        # Targets
        meta = sample["metadata"]
        EF_list.append(meta["EF"])

        # Feature extraction (same as your original)
        z_proj = savgol_filter(project_to_major_axis(z, phase, axis=EDES_axis), window_length=11, polyorder=3, axis=0)
        peaks, valleys = find_peaks_sentinel(z_proj, p=0.2, d=5)

        # grid, mu = von_mises_kernel_smoother(z, phase, n_grid=512, kappa=30)

        # peak_phase = np.mean(phase[peaks])
        # valley_phase = np.mean(phase[valleys])

        # peak_idx = int(np.argmin(np.abs(grid - peak_phase)))
        # valley_idx = int(np.argmin(np.abs(grid - valley_phase)))

        # A = mu[peak_idx]; B = mu[valley_idx]
        # feature = np.concatenate([A, B], axis=0)
        # feature = np.array([np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(A-B)])
        feature = np.mean(z[peaks], axis=0) - np.mean(z[valleys], axis=0)
        features_list.append(feature)

    X = np.asarray(features_list)
    y = np.asarray(EF_list)
    return X, y


def eval_reg(name, reg, X, y):
    y_pred = reg.predict(X)
    mae = float(np.mean(np.abs(y - y_pred)))
    r2 = float(reg.score(X, y))
    print(f"{name} | MAE: {mae:.4f} | R^2: {r2:.4f}")
    return mae, r2


# Load data
train_ds, val_ds, test_ds = get_latents_dataset()

# Build train / val features
X_train, y_train = build_X_y(train_ds, desc="train")
X_val, y_val = build_X_y(test_ds, desc="val")

# Fit on train
reg_EF = Ridge(alpha=1.0).fit(X_train, y_train)

print("\n== Train metrics ==")
eval_reg("EF ", reg_EF, X_train, y_train)

print("\n== Val metrics ==")
eval_reg("EF ", reg_EF, X_val, y_val)