# sample_mask_pediatric.py
#
# Pick the first traced video in each view (A4C + PSAX), rasterize the tracing polygon
# for one frame, and save a 3-panel sanity-check image:
#   [echo frame] [mask] [overlay (blue mask)]
#
# Usage:
#   python sample_mask_pediatric.py --data_root ./data/echonetpediatric --out_dir ./debug_masks
#
# Outputs:
#   ./debug_masks/sample_mask_A4C.png
#   ./debug_masks/sample_mask_PSAX.png

import os
import csv
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import torch
from torchvision.io import read_video

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def load_tracings_polygons_xy(tracings_csv_path: str) -> Dict[str, Dict[int, List[Tuple[float, float]]]]:
    """
    Pediatric VolumeTracings.csv columns: FileName,X,Y,Frame
    Returns: tracings["XXXX.avi"][frame] = [(x,y), ...] in CSV row order.
    """
    tracings: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
    with open(tracings_csv_path, newline="") as f:
        r = csv.DictReader(f)
        expected = {"FileName", "X", "Y", "Frame"}
        cols = set(r.fieldnames or [])
        if not expected.issubset(cols):
            raise ValueError(f"Expected columns {sorted(expected)}, got {sorted(cols)}")

        for row in r:
            try:
                fn = row["FileName"].strip()
                frame = int(row["Frame"])
                x = float(row["X"])
                y = float(row["Y"])
                tracings.setdefault(fn, {}).setdefault(frame, []).append((x, y))
            except:
                continue
    return tracings


def rasterize_polygon(points_xy: List[Tuple[int, int]], hw: Tuple[int, int]) -> torch.Tensor:
    """Return uint8 mask [H,W] with {0,1}."""
    H, W = hw
    img = Image.new("L", (W, H), 0)
    if len(points_xy) >= 3:
        ImageDraw.Draw(img).polygon(points_xy, outline=1, fill=1)
    buf = bytearray(img.tobytes())
    return (torch.frombuffer(buf, dtype=torch.uint8).view(H, W) > 0).to(torch.uint8)


def points_to_mask(pts: List[Tuple[float, float]], hw: Tuple[int, int]) -> torch.Tensor:
    """pts are (x,y) in pixel coords; clamp to image bounds and rasterize."""
    H, W = hw
    if len(pts) < 3:
        return torch.zeros((H, W), dtype=torch.uint8)

    pts_i: List[Tuple[int, int]] = []
    for x, y in pts:
        xi = max(0, min(W - 1, int(round(x))))
        yi = max(0, min(H - 1, int(round(y))))
        pts_i.append((xi, yi))
    return rasterize_polygon(pts_i, (H, W))


def pick_first_traced_example(tracings: Dict[str, Dict[int, List[Tuple[float, float]]]]) -> Tuple[str, int]:
    """
    Returns (filename_with_ext, frame_idx).
    Picks lexicographically first filename, then smallest frame.
    """
    if not tracings:
        raise RuntimeError("No tracings found.")
    fn = sorted(tracings.keys())[0]
    frames = sorted(tracings[fn].keys())
    if not frames:
        raise RuntimeError(f"No frames found for {fn}")
    return fn, int(frames[0])


def load_video_frame(video_path: str, frame_idx: int) -> Tuple[torch.Tensor, float]:
    """
    Returns (frame_rgb_u8 [H,W,3], fps).
    Uses read_video; frame_idx is integer index in decoded tensor.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
        )
        v, _, info = read_video(video_path, pts_unit="sec")  # v: [T,H,W,C], uint8-ish
    if v.numel() == 0:
        raise RuntimeError(f"Decoded empty video: {video_path}")

    T = int(v.shape[0])
    if frame_idx < 0 or frame_idx >= T:
        raise RuntimeError(f"Requested frame {frame_idx}, but T={T} for {video_path}")

    # frame to uint8 cpu
    frame = v[frame_idx].contiguous().cpu()
    if frame.dtype != torch.uint8:
        frame = frame.to(torch.uint8)

    fps = 0.0
    try:
        if isinstance(info, dict) and info.get("video_fps") is not None:
            fps = float(info["video_fps"])
    except Exception:
        pass

    return frame, fps


def save_triptych(frame_rgb_u8: torch.Tensor, mask_u8: torch.Tensor, out_path: str, title: str):
    """
    frame_rgb_u8: [H,W,3] uint8
    mask_u8:      [H,W] uint8 in {0,1}
    Saves 1x3 image: frame | mask | overlay(blue)
    """
    frame_np = frame_rgb_u8.numpy()
    mask_np = mask_u8.numpy()

    H, W = mask_np.shape
    if frame_np.shape[0] != H or frame_np.shape[1] != W:
        raise ValueError("Frame/mask size mismatch")

    # Build blue overlay
    overlay = frame_np.copy()
    # add blue where mask==1 (simple alpha blend)
    alpha = 0.45
    blue = (0, 0, 255)
    m = mask_np.astype(bool)
    overlay[m, 0] = (1 - alpha) * overlay[m, 0] + alpha * blue[0]
    overlay[m, 1] = (1 - alpha) * overlay[m, 1] + alpha * blue[1]
    overlay[m, 2] = (1 - alpha) * overlay[m, 2] + alpha * blue[2]
    overlay = overlay.astype("uint8")

    # Plot
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(frame_np)
    ax1.set_title("Echo frame")
    ax1.axis("off")

    ax2.imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Mask")
    ax2.axis("off")

    ax3.imshow(overlay)
    ax3.set_title("Overlay (blue)")
    ax3.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_for_view(data_root: str, view: str, out_dir: str):
    view_root = os.path.join(data_root, view)
    videos_dir = os.path.join(view_root, "Videos")
    tracings_csv = os.path.join(view_root, "VolumeTracings.csv")

    if not os.path.isfile(tracings_csv):
        raise FileNotFoundError(tracings_csv)
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(videos_dir)

    tracings = load_tracings_polygons_xy(tracings_csv)
    fn, frame_idx = pick_first_traced_example(tracings)

    video_path = os.path.join(videos_dir, fn)
    if not os.path.isfile(video_path):
        # some datasets might store without extension mismatch; be explicit
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame, fps = load_video_frame(video_path, frame_idx)
    H, W = int(frame.shape[0]), int(frame.shape[1])

    pts = tracings[fn][frame_idx]
    mask = points_to_mask(pts, (H, W))  # uint8 {0,1}

    out_path = os.path.join(out_dir, f"sample_mask_{view}.png")
    title = f"{view} | {fn} | frame={frame_idx} | fps={fps:.2f} | n_pts={len(pts)} | area={int(mask.sum().item())}"
    save_triptych(frame, mask, out_path, title)
    print(f"[{view}] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data/echonetpediatric")
    ap.add_argument("--out_dir", type=str, default="./debug_masks")
    args = ap.parse_args()

    for view in ("A4C", "PSAX"):
        run_for_view(args.data_root, view, args.out_dir)


if __name__ == "__main__":
    main()