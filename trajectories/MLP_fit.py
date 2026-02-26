import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
from scipy.signal import detrend

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from Dataset import get_latents_dataset
from utils.topology import project_to_major_axis, von_mises_kernel_smoother
from tasks.Compute_EDES import find_peaks_sentinel

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
        z_proj = project_to_major_axis(z, phase, axis=EDES_axis)  # [T]
        peaks, valleys = find_peaks_sentinel(z_proj, p=0.2, d=5)

        grid, mu = von_mises_kernel_smoother(z, phase, n_grid=512, kappa=30)

        peak_phase = np.mean(phase[peaks])
        valley_phase = np.mean(phase[valleys])

        peak_idx = int(np.argmin(np.abs(grid - peak_phase)))
        valley_idx = int(np.argmin(np.abs(grid - valley_phase)))

        A = mu[peak_idx]
        B = mu[valley_idx]
        feature = np.concatenate([A, B], axis=0)
        features_list.append(feature)

    X = np.asarray(features_list, dtype=np.float32)
    y = {
        "EF": np.asarray(EF_list, dtype=np.float32),
    }
    return X, y


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim, out_dim: int = 1):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, out_dim))
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


def denorm(x_norm_t: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x_norm_t * std + mean


@torch.no_grad()
def eval_mae_and_std_native(model, loader, device, mean, std):
    """
    Computes MAE and std(|error|) in *native* target space
    even though the model predicts in normalized space.
    """
    model.eval()
    abs_errs_native = []
    for xb, yb_norm in loader:
        xb = xb.to(device)
        yb_norm = yb_norm.to(device)

        pred_norm = model(xb).squeeze(-1)

        pred_native = denorm(pred_norm, mean, std)
        y_native = denorm(yb_norm, mean, std)

        abs_err_native = (pred_native - y_native).abs()
        abs_errs_native.append(abs_err_native.detach().cpu())

    abs_errs_native = torch.cat(abs_errs_native, dim=0)
    mae = float(abs_errs_native.mean().item())
    std_abs = float(abs_errs_native.std(unbiased=False).item())
    return mae, std_abs


def train_one_target(
    name,
    X_train,
    y_train_norm,
    X_val,
    y_val_norm,
    mean,
    std,
    epochs=200,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-3,
    hidden_dim=512,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train_norm)
    Xva = torch.from_numpy(X_val)
    yva = torch.from_numpy(y_val_norm)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False, drop_last=False)

    model = MLPRegressor(in_dim=X_train.shape[1], hidden_dim=hidden_dim, out_dim=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {name} | epoch {ep}/{epochs}", leave=False)
        running = 0.0
        n = 0

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            running += float(loss.item()) * bs
            n += bs
            pbar.set_postfix(loss=running / max(n, 1))
        scheduler.step()

    # Final metrics in native space
    train_mae, train_std_abs = eval_mae_and_std_native(model, train_loader, device, mean, std)
    val_mae, val_std_abs = eval_mae_and_std_native(model, val_loader, device, mean, std)

    print(f"{name} | Train MAE: {train_mae:.4f} | STD(|err|): {train_std_abs:.4f}")
    print(f"{name} |   Val MAE: {val_mae:.4f} | STD(|err|): {val_std_abs:.4f}")

    return model


# -------------------------
# Main
# -------------------------
train_ds, val_ds, test_ds = get_latents_dataset()

X_train, y_train = build_X_y(train_ds, desc="train")
X_val, y_val = build_X_y(val_ds, desc="val")

# Normalize targets using TRAIN mean/std (per-target)
stats = {}
y_train_norm = {}
y_val_norm = {}

for k in ["EF"]:
    mu = float(np.mean(y_train[k]))
    sd = float(np.std(y_train[k]) + 1e-8)
    stats[k] = (mu, sd)
    y_train_norm[k] = ((y_train[k] - mu) / sd).astype(np.float32)
    y_val_norm[k] = ((y_val[k] - mu) / sd).astype(np.float32)

print("\nTarget normalization stats (train):")
for k, (mu, sd) in stats.items():
    print(f"  {k}: mean={mu:.6f}, std={sd:.6f}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}\n")

model_EF  = train_one_target("EF ", X_train, y_train_norm["EF"],  X_val, y_val_norm["EF"],  *stats["EF"],  device=device)