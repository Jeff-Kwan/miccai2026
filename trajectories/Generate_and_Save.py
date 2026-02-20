import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from datahandling.collate import Traj_collate
from models.SplineAutoEncoder import SplineAutoEncoder
import os
import numpy as np
from tqdm import tqdm
import json
from math import ceil
from utils.topology import laplacian_phase


def run_split(dl, split_name: str, autocast: bool):
    ed_mae_list, es_mae_list, fps_all = [], [], []

    with torch.inference_mode():
        for idx, batch in tqdm(enumerate(dl), desc=split_name):
            videos = batch['video']  # [B, T, C, H, W]
            timestamps = batch['timestamps']  # [B, T]
            fps = batch['fps']  # [B]
            frames_idx = batch['frame_indices']  # [B, 2]
            metadata = batch['metadata'][0]
            gt_es, gt_ed = frames_idx[0]
            fps = float(fps[0])
            gt_es = int(gt_es.item())
            gt_ed = int(gt_ed.item())

            videos = videos.to(device, non_blocking=True)
            timestamps = timestamps.to(device, non_blocking=True)

            dense_factor = 4
            t0 = timestamps.min(); t1 = timestamps.max(); T = timestamps.shape[1]
            dense_t = torch.linspace(t0, t1, (T-1)*dense_factor+1, device=device).unsqueeze(0)

            with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                z = model.encode(videos)
            z_spline = model.spline_fit_and_eval(z, timestamps, dense_t)

            # Move to CPU numpy
            z_np = z.squeeze(0).cpu().numpy()
            z_spline_np = z_spline.squeeze(0).cpu().numpy()
            timestamps_np = timestamps.squeeze(0).cpu().numpy()

            phase = laplacian_phase(z_spline_np)[0]

            z_spline_np = z_spline_np[::dense_factor]
            phase = phase[::dense_factor]

            traj_data = {
                "z": z_np,
                "z_spline": z_spline_np,
                "timestamps": timestamps_np,
                "phase": phase,
                "frame_indices": (gt_es, gt_ed),
                "fps": fps,
                "metadata": metadata,
            }

            # Save to /data/echodyna/latents
            out_dir = f"/data/echodyna/latents/{split_name}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{idx:04d}.pt")
            torch.save(traj_data, out_path)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dir = "results/2026_02_19/16_20_SAE"

    print("Starting ED/ES evaluation...")

    # ---- Model ----
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
    model = model.to(device).eval()

    # ---- Dataset ----
    train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=Traj_collate,
        num_workers=24,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=Traj_collate,
        num_workers=24,
        pin_memory=True
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=Traj_collate,
        num_workers=24,
        pin_memory=True
    )

    # ---- Evaluation ----
    dls = [train_dl, val_dl, test_dl]
    for dl, split_name in zip(dls, ["Train", "Val", "Test"]):
        run_split(dl, split_name, autocast)
