import os, csv, warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video


# =========================
# Dataset + CSV loading
# =========================

@dataclass
class EchoDynaItem:
    filename: str
    metadata: Dict[str, Any]


class EchoDynaDataset(Dataset):
    def __init__(self, entries: List[EchoDynaItem], video_backend: Optional[str] = None):
        self.entries, self.video_backend = entries, video_backend

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]

        # NOTE: torchvision deprecated warnings can be noisy
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
            )
            v, _, _ = read_video(e.filename, pts_unit="sec")  # [T,H,W,C] uint8-ish

        v = v.permute(0, 3, 1, 2).float() / 255.0 * 2 - 1  # [T,C,H,W] in [-1,1]
        return {"filename": e.filename, "metadata": dict(e.metadata), "video": v}


REQ = {
    "FileName",
    "EF",
    "ESV",
    "EDV",
    "FrameHeight",
    "FrameWidth",
    "FPS",
    "NumberOfFrames",
    "Split",
}
META_CAST = {
    "EF": float,
    "ESV": float,
    "EDV": float,
    "FrameHeight": int,
    "FrameWidth": int,
    "FPS": float,
    "NumberOfFrames": int,
}


def load_echonet_dynamic_datasets(
    csv_path: str,
    videos_dir: str,
    video_backend: Optional[str] = None,
    verify_files_exist: bool = False,
) -> Tuple[EchoDynaDataset, EchoDynaDataset, EchoDynaDataset]:
    splits = {"TRAIN": [], "VAL": [], "TEST": []}

    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        missing = REQ - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in r:
            fn = os.path.join(videos_dir, row["FileName"] + ".avi")
            if verify_files_exist and not os.path.isfile(fn):
                raise FileNotFoundError(f"Missing video file: {fn}")

            meta = {k: META_CAST[k](row[k]) for k in META_CAST}
            key = row["Split"].strip().upper()
            if key not in splits:
                raise ValueError(f"Unknown split value: {row['Split']}")
            splits[key].append(EchoDynaItem(fn, meta))

    mk = lambda xs: EchoDynaDataset(xs, video_backend=video_backend)
    return mk(splits["TRAIN"]), mk(splits["VAL"]), mk(splits["TEST"])


# =========================
# Resampling helpers
# =========================

def resample_fps(video: torch.Tensor, src_fps: float, dst_fps: float, mode: str = "nearest"):
    """
    video: [T,C,H,W]
    returns: [T',C,H,W]
    """
    if video.ndim != 4:
        raise ValueError("Expected [T,C,H,W]")
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("FPS must be positive")

    T = video.size(0)
    if T == 0 or abs(src_fps - dst_fps) < 1e-6:
        return video

    new_T = max(1, int(round((T / src_fps) * dst_fps)))
    pos = torch.linspace(0, T - 1, new_T, device=video.device)

    if mode == "nearest":
        return video[pos.round().long().clamp(0, T - 1)]
    if mode == "linear":
        li = pos.floor().long().clamp(0, T - 1)
        ri = (li + 1).clamp(0, T - 1)
        a = (pos - li).view(-1, 1, 1, 1)
        return (1 - a) * video[li] + a * video[ri]

    raise ValueError("mode must be 'nearest' or 'linear'")


def resample_resolution(video: torch.Tensor, target=(128, 128), mode="bilinear"):
    """
    video: [T,C,H,W]
    returns: [T,C,th,tw]
    """
    if video.ndim != 4:
        raise ValueError("Expected [T,C,H,W]")
    th, tw = (target, target) if isinstance(target, int) else target
    if (th <= 0) or (tw <= 0):
        raise ValueError("target must be positive")
    if video.shape[-2:] == (th, tw):
        return video

    # align_corners only valid for certain modes
    ac = False if mode in {"bilinear", "bicubic", "trilinear"} else None
    return F.interpolate(video, size=(th, tw), mode=mode, align_corners=ac)


# =========================
# Drop-in replacement: preprocess + saving, with UPDATED metadata
# =========================

def collate_fn(b):  # batch_size=1
    x = b[0]
    return x["video"], x["metadata"], x["filename"]


@torch.no_grad()
def preprocess_video(
    video: torch.Tensor,
    src_fps: float,
    dst_fps: float = 24.0,
    size=(128, 128),
    fps_mode: str = "nearest",
    resize_mode: str = "bilinear",
):
    """
    video: [T,C,H,W] in [-1,1]
    returns: [T',C,h,w] in [-1,1]
    """
    v = resample_fps(video, src_fps, dst_fps, mode=fps_mode)
    v = resample_resolution(v, target=size, mode=resize_mode)
    return v


def _updated_metadata_after_preprocess(
    meta: Dict[str, Any],
    *,
    src_video: torch.Tensor,
    dst_video: torch.Tensor,
    dst_fps: float,
) -> Dict[str, Any]:
    """
    Returns a NEW metadata dict with:
      - original fields preserved (also mirrored as Original*)
      - updated FPS / NumberOfFrames / FrameHeight / FrameWidth matching dst_video
    """
    out = dict(meta)

    # Preserve originals explicitly (handy when debugging / training)
    # Only add if not already present.
    def keep_orig(k: str, v: Any):
        ok = f"Original{k}"
        if ok not in out:
            out[ok] = v

    keep_orig("FPS", float(out.get("FPS", dst_fps)))
    keep_orig("NumberOfFrames", int(out.get("NumberOfFrames", int(src_video.shape[0]))))
    keep_orig("FrameHeight", int(out.get("FrameHeight", int(src_video.shape[-2]))))
    keep_orig("FrameWidth", int(out.get("FrameWidth", int(src_video.shape[-1]))))

    # Now overwrite with post-resample truth
    out["FPS"] = float(dst_fps)
    out["NumberOfFrames"] = int(dst_video.shape[0])           # T'
    out["FrameHeight"] = int(dst_video.shape[-2])             # h
    out["FrameWidth"] = int(dst_video.shape[-1])              # w

    return out


@torch.no_grad()
def save_preprocessed_split(
    dl: DataLoader,
    split_name: str,
    out_root: str = "data/echodyna/preprocessed",
    dst_fps: float = 24.0,
    size=(128, 128),
    overwrite: bool = False,
    dtype: torch.dtype = torch.float16,
    fps_mode: str = "nearest",
    resize_mode: str = "bilinear",
    save_ext: str = ".pt",
):
    """
    Drop-in replacement:
      - saves {"video": [C,T,h,w], "metadata": UPDATED_META, ...}
      - metadata fields updated to match resampling:
          FPS, NumberOfFrames, FrameHeight, FrameWidth
      - also keeps OriginalFPS/OriginalNumberOfFrames/OriginalFrameHeight/OriginalFrameWidth
    """
    split_dir = os.path.join(out_root, split_name.upper())
    os.makedirs(split_dir, exist_ok=True)

    n_ok, n_skip, n_fail = 0, 0, 0

    for video, meta, filename in tqdm(dl, desc=f"Preprocessing {split_name}", unit="video"):
        video = video.squeeze(0)  # [T,C,H,W]
        meta = dict(meta)         # make sure it's mutable + detached from dataset item

        base = os.path.splitext(os.path.basename(filename))[0]
        out_path = os.path.join(split_dir, base + save_ext)

        if (not overwrite) and os.path.isfile(out_path):
            n_skip += 1
            continue

        try:
            # Prefer actual tensor length as "truth" if CSV is off,
            # but still use CSV FPS for temporal scaling.
            src_fps = float(meta.get("FPS", 0.0))
            if src_fps <= 0:
                raise ValueError(f"Bad src FPS in metadata: {meta.get('FPS')}")

            v = preprocess_video(
                video,
                src_fps=src_fps,
                dst_fps=dst_fps,
                size=size,
                fps_mode=fps_mode,
                resize_mode=resize_mode,
            )  # [T',C,h,w]

            # Update metadata to reflect resampling results
            meta_upd = _updated_metadata_after_preprocess(
                meta,
                src_video=video,
                dst_video=v,
                dst_fps=dst_fps,
            )

            # Store as [C,T,h,w]
            v = v.permute(1, 0, 2, 3).contiguous()
            v = v.to(dtype=dtype).cpu()

            payload = {
                "video": v,
                "fps": float(dst_fps),
                "size": tuple(size) if not isinstance(size, int) else (int(size), int(size)),
                "source_path": filename,
                "metadata": meta_upd,  # <-- UPDATED + ORIGINALS preserved
            }

            tmp_path = out_path + ".tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, out_path)

            n_ok += 1

        except Exception as e:
            n_fail += 1
            print(f"\n[{split_name}] FAILED: {filename}\n  -> {e}")

    print(f"[{split_name}] done: ok={n_ok}, skipped={n_skip}, failed={n_fail}")


# =========================
# Usage (same as before)
# =========================

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(
        "data/echodyna/FileList.csv", "data/echodyna/Videos"
    )

    workers = 62
    train_dl = DataLoader(train_ds, batch_size=1, num_workers=workers, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=1, num_workers=workers, collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=1, num_workers=workers, collate_fn=collate_fn)

    save_preprocessed_split(train_dl, "TRAIN", out_root="data/echodyna/preprocessed", dst_fps=24.0, size=(128, 128))
    save_preprocessed_split(val_dl,   "VAL",   out_root="data/echodyna/preprocessed", dst_fps=24.0, size=(128, 128))
    save_preprocessed_split(test_dl,  "TEST",  out_root="data/echodyna/preprocessed", dst_fps=24.0, size=(128, 128))
