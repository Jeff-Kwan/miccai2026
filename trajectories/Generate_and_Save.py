import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from datahandling.collate import Traj_collate
from models.SplineAutoEncoder import SplineAutoEncoder
from utils.topology import laplacian_phase
import os
from tqdm import tqdm
import json
import shutil


def run_split(dl, split_name: str, autocast: bool):
    out_dir = f"/data/echodyna/latents/{split_name}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Overwriting output directory {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    with torch.inference_mode():
        for idx, batch in tqdm(enumerate(dl), desc=split_name):
            video = batch["video"].unsqueeze(0).to(device, non_blocking=True)  # [1,T,C,H,W]
            gt_ed = batch["ED"]
            gt_es = batch["ES"]
            fps = batch["fps"]
            metadata = batch["metadata"]

            with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
                z = model.encode(video)

            # Move to CPU numpy
            z_np = z.squeeze(0).cpu().numpy()
            phase = laplacian_phase(z_np)[0]

            traj_data = {
                "z": z_np,
                "phase": phase,
                "ED": gt_ed,
                "ES": gt_es,
                "fps": fps,
                "metadata": metadata,
            }
            out_path = os.path.join(out_dir, f"{idx:04d}.pt")
            torch.save(traj_data, out_path)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dir = "results/2026_02_24/14_57_SAE"

    # ---- Model ----
    config = json.load(open("config/SAE.json", "r"))
    mcfg = config["model"]
    model = SplineAutoEncoder(
        latent=config["model"]["latent"],
        in_dim=config["model"].get("in_dim", 3),
        out_dim=config["model"].get("out_dim", None),
        n_ctrl_params=config["model"]["n_ctrl_params"],
        degree=config["model"]["degree"],
        lam=config["model"]["lam"],
        ).to(device)

    model.load_state_dict(torch.load(os.path.join(load_dir, "SAE.pth"), map_location=device))
    autocast = config["training"].get("autocast", False)
    model = model.to(device).eval()

    # ---- Dataset ----
    train_ds, val_ds, test_ds = load_echonet_dynamic_datasets()

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
    dls = [test_dl]
    for dl, split_name in zip(dls, ["Test"]):
        run_split(dl, split_name, autocast)
