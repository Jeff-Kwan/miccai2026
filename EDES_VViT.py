import torch
from torch.utils.data import DataLoader
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from models.VideoViT2 import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg
from models.ViTMAEMotion2 import VideoMotionMAE, SimpleConvDecoder
import os
from ref_utils.LMP.LMP_utils import compute_main_orientation_and_extrema
import numpy as np
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_12/16_08_VMAE"
autocast = True

# ---- Model ----
config = json.load(open("config/VMAE.json", "r"))
enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
dec = VideoViTDecoder(
    enc_dim=config["encoder"]["dim"],
    patch=config["encoder"]["patch"],
    in_chans=config["encoder"]["in_chans"],
    cfg=VideoViTDecCfg(**config["decoder"]),
)
frame_dec = SimpleConvDecoder(
    latent=config["encoder"]["dim"],
    out_dim=config["encoder"]["in_chans"],
    base=config["decoder"]["dec_dim"],
)
mae = VideoMotionMAE(enc, dec, frame_dec, motion_dim=2, norm_pix_loss=False, mask_ratio=0.75)
mae.load_state_dict(torch.load(os.path.join(load_dir, "VMAE.pth"), map_location=device))
mae = mae.to(device).eval()

# ---- Dataset ----
def collate_fn(batch):
    videos = torch.stack([sample["video"] for sample in batch], dim=0)          # [B,T,C,H,W]
    timestamps = torch.stack([sample["timestamps"] for sample in batch], dim=0) # [B,T]

    frames_idx = []; fps_list = []
    for sample in batch:
        fi = sample["masks"]["frame_indices"]
        if fi is None:
            fi = torch.empty((0,), dtype=torch.long)
        else:
            fi = fi.long()
        frames_idx.append(fi)
        fps_list.append(sample["metadata"]["FPS"])
        
    return videos, timestamps, frames_idx, fps_list

train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=True)

train_dl = DataLoader(
    train_ds, batch_size=1, shuffle=True,
    collate_fn=collate_fn, num_workers=32,
    pin_memory=True, persistent_workers=True
)
val_dl = DataLoader(
    val_ds, batch_size=1, shuffle=True,
    collate_fn=collate_fn, num_workers=32, pin_memory=True
)
test_dl = DataLoader(
    test_ds, batch_size=1, shuffle=False,
    collate_fn=collate_fn, num_workers=32, pin_memory=True
)

def eval_split(dl, split_name: str, use_amp: bool = True):
    ed_mae_list, es_mae_list, fps_all = [], [], []

    with torch.inference_mode():
        for videos, timestamps, frames_idx, fps in tqdm(dl, desc=split_name):
            # [0] because we go through one by one (variable length video)
            gt_es, gt_ed = frames_idx[0]
            fps = float(fps[0])
            gt_es = int(gt_es.item())
            gt_ed = int(gt_ed.item())

            videos = videos.to(device, non_blocking=True)
            timestamps = timestamps.to(device, non_blocking=True)

            with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
                B, T, C, H, W = videos.shape
                N = (H // mae.encoder.cfg.patch) * (W // mae.encoder.cfg.patch)
                # No masking for encoder, keep all indices
                keep_idx = torch.arange(N, device=device)[None, None, :].expand(B, T, N)
                gcls, enc_tokens, _ = mae.encoder(videos, keep_idx=keep_idx, timestamps=timestamps)
                frame_cls = enc_tokens[:, :, 0, :]
                _, z_motion = mae.low_rank_latent(gcls, frame_cls, T)
                
            z_motion = z_motion.float().squeeze().cpu().numpy()

            group_ed, group_es, edpoint, espoint, traj, direction = \
                compute_main_orientation_and_extrema(z_motion, fps, visualize=False)

            # Mean Absolute Error
            ed_err = min(abs(edpt - gt_ed) for edpt in group_ed)
            es_err = min(abs(espt - gt_es) for espt in group_es)

            ed_mae_list.append(ed_err)
            es_mae_list.append(es_err)
            fps_all.append(fps)


    mean_ed_mae = float(np.mean(ed_mae_list)) if len(ed_mae_list) else float("nan")
    mean_es_mae = float(np.mean(es_mae_list)) if len(es_mae_list) else float("nan")

    # Convert per-frame mae to ms using sample-FPS
    ed_ms = [mae * (1000.0 / fps) for mae, fps in zip(ed_mae_list, fps_all)]
    es_ms = [mae * (1000.0 / fps) for mae, fps in zip(es_mae_list, fps_all)]
    mean_ed_ms = np.mean(ed_ms) if len(ed_ms) else float("nan")
    mean_es_ms = np.mean(es_ms) if len(es_ms) else float("nan")

    # "Same format as now"
    out_lines = [
        f"{split_name} ED MAE: {mean_ed_mae:.2f} frames, {mean_ed_ms:.2f} ms",
        f"{split_name} ES MAE: {mean_es_mae:.2f} frames, {mean_es_ms:.2f} ms\n",
    ]

    print(out_lines[0])
    print(out_lines[1])

    return {
        "split": split_name,
        "ed_frames": mean_ed_mae,
        "ed_ms": mean_ed_ms,
        "es_frames": mean_es_mae,
        "es_ms": mean_es_ms,
        "lines": out_lines
    }

results = []
results.append(eval_split(train_dl, "Train", use_amp=autocast))
results.append(eval_split(val_dl, "Val", use_amp=autocast)) 
results.append(eval_split(test_dl, "Test", use_amp=autocast))

# ---- Save to <load_dir>/edes_detection.txt ----
out_path = os.path.join(load_dir, "edes_detection.txt")
with open(out_path, "w") as f:
    for r in results:
        f.write(r["lines"][0] + "\n")
        f.write(r["lines"][1] + "\n")

print(f"\nSaved results to: {out_path}")