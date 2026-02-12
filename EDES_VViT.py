import torch 
from torch import nn
from torch.utils.data import DataLoader
from datahandling.PreTrainEchoDynaDataset2 import load_echodyna_downstream_datasets
from models.VideoViT2 import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg
from models.ViTMAEMotion2 import VideoMotionMAE, SimpleConvDecoder
import os
from ref_utils.LMP.LMP_utils import compute_main_orientation_and_extrema
from scipy.signal import find_peaks, savgol_filter
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dir = "results/2026_02_11/17_24_VMAE"

config = json.load(open("config/VMAE.json", "r"))
enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
dec = VideoViTDecoder(enc_dim=config["encoder"]["dim"], patch=config["encoder"]["patch"], 
                      in_chans=config["encoder"]["in_chans"], cfg=VideoViTDecCfg(**config["decoder"]))
frame_dec = SimpleConvDecoder(latent=config["encoder"]["dim"], out_dim=config["encoder"]["in_chans"], base=config["decoder"]["dec_dim"])
mae = VideoMotionMAE(enc, dec, frame_dec, motion_dim=2, norm_pix_loss=False, mask_ratio=0.75)
mae.load_state_dict(torch.load(os.path.join(load_dir, "VMAE.pth"), map_location=device))
mae = mae.to(device)
mae.eval()

# Dataset
def collate_fn(batch):
    # stack videos: [B, T, C, H, W]
    videos = torch.stack([sample["video"] for sample in batch], dim=0)
    timestamps = torch.stack([sample["timestamps"] for sample in batch], dim=0)

    # collect frame indices
    frames_idx = []
    for sample in batch:
        fi = sample.get("frame_indices", None)
        if fi is None:
            fi = torch.empty((0,), dtype=torch.long)
        else:
            fi = fi.long()
        frames_idx.append(fi)

    return videos, timestamps, frames_idx

train_ds, val_ds, test_ds = load_echodyna_downstream_datasets(allow_missing_masks=False)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=60, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=32, pin_memory=True,
                    collate_fn=collate_fn)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=32, pin_memory=True,
                     collate_fn=collate_fn)



def detect_peaks_and_valleys(
    sequence,
    smooth=True,
    window_length=9,
    polyorder=2,
    prominence_ratio=0.3,
    edge_events=False
):
    seq = np.asarray(sequence)

    # Optional smoothing
    if smooth and len(seq) >= window_length:
        processed = savgol_filter(seq, window_length, polyorder)
    else:
        processed = seq.copy()

    # Prominence threshold based on signal range
    prominence_threshold = prominence_ratio * (np.max(processed) - np.min(processed))

    # Handle edge extrema if requested
    if edge_events:
        peak_input = np.concatenate(([np.min(processed)], processed, [np.min(processed)]))
        valley_input = np.concatenate(([np.max(processed)], processed, [np.max(processed)]))
    else:
        peak_input = processed
        valley_input = processed

    # Detect peaks and valleys
    peaks, _ = find_peaks(peak_input, prominence=prominence_threshold)
    valleys, _ = find_peaks(-valley_input, prominence=prominence_threshold)

    # Adjust indices if padding was used
    if edge_events:
        peaks = peaks - 1
        valleys = valleys - 1

        # Remove out-of-range indices caused by padding
        peaks = peaks[(peaks >= 0) & (peaks < len(seq))]
        valleys = valleys[(valleys >= 0) & (valleys < len(seq))]

    return peaks, valleys, processed


ed_mae_list = []
es_mae_list = []
with torch.inference_mode():
    for videos, timestamps, frames_idx in tqdm(test_dl):
        # pull scalars with minimal indexing + conversions
        gt_es, gt_ed = frames_idx[0]          # (es, ed) tensors
        gt_es = gt_es.item()
        gt_ed = gt_ed.item()

        # forward + convert once
        videos = videos.to(device)  # [1, T, C, H, W]
        timestamps = timestamps.to(device)  # [1, T]
        with torch.autocast('cuda', torch.bfloat16):
            z_motion = mae(videos, timestamps)["z_motion"]
        z_motion = z_motion.float().squeeze().cpu().numpy()

        group_ed, group_es, edpoint, espoint, traj, direction = \
            compute_main_orientation_and_extrema(z_motion, 24, visualize=False)
        # group_es, group_ed, _ = detect_peaks_and_valleys(z_motion[:, 1], edge_events=True)

        # no intermediate lists
        ed_mae_list.append(min(abs(edpt - gt_ed) for edpt in group_ed))
        es_mae_list.append(min(abs(espt - gt_es) for espt in group_es))

mean_ed_mae = np.mean(ed_mae_list)
mean_es_mae = np.mean(es_mae_list)
print(f"ED MAE: {mean_ed_mae:.2f} frames, {mean_ed_mae*1000/24:.2f} ms")
print(f"ES MAE: {mean_es_mae:.2f} frames, {mean_es_mae*1000/24:.2f} ms")