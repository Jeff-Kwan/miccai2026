import os, csv, warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from PIL import Image, ImageDraw


# =========================
# Dataset + CSV loading
# =========================

REQ = {"FileName", "EF", "ESV", "EDV", "FrameHeight", "FrameWidth", "FPS", "NumberOfFrames", "Split"}
META_CAST = {
    "EF": float, "ESV": float, "EDV": float,
    "FrameHeight": int, "FrameWidth": int,
    "FPS": float, "NumberOfFrames": int,
}

@dataclass(frozen=True)
class EchoDynaItem:
    filename: str
    metadata: Dict[str, Any]

class EchoDynaDataset(Dataset):
    def __init__(self, entries: List[EchoDynaItem], video_backend: Optional[str] = None):
        self.entries = entries
        if video_backend:
            try:
                from torchvision import set_video_backend
                set_video_backend(video_backend)
            except Exception:
                pass  # best-effort

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        e = self.entries[i]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
            )
            v, _, _ = read_video(e.filename, pts_unit="sec")  # [T,H,W,C]

        v = v.permute(0, 3, 1, 2).float().div_(255).mul_(2).sub_(1)  # [T,C,H,W] in [-1,1]
        return {"filename": e.filename, "metadata": dict(e.metadata), "video": v}

def load_echonet_dynamic_datasets(
    csv_path: str,
    videos_dir: str,
    video_backend: Optional[str] = None,
    verify_files_exist: bool = False,
) -> Tuple[EchoDynaDataset, EchoDynaDataset, EchoDynaDataset]:
    splits: Dict[str, List[EchoDynaItem]] = {"TRAIN": [], "VAL": [], "TEST": []}

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
            split = row["Split"].strip().upper()
            if split not in splits:
                raise ValueError(f"Unknown split value: {row['Split']}")
            splits[split].append(EchoDynaItem(fn, meta))

    mk = lambda xs: EchoDynaDataset(xs, video_backend=video_backend)
    return mk(splits["TRAIN"]), mk(splits["VAL"]), mk(splits["TEST"])


# =========================
# VolumeTracings loading (FIXED)
# =========================

TRACING_REQ = {"FileName", "X1", "Y1", "X2", "Y2", "Frame"}

def load_volume_tracings(tracings_csv_path: str) -> Dict[str, Dict[int, List[Tuple[float, float]]]]:
    """
    Correct interpretation for EchoNet-Dynamic VolumeTracings.csv:
    each row describes a boundary *segment* (X1,Y1)->(X2,Y2). The rows are ordered such that
    taking the sequence of (X1,Y1) gives the polygon vertices around the contour.

    The previous logic that appended both endpoints produces duplicated/zig-zag point lists,
    which yields incorrect filled masks.
    """
    tracings: Dict[str, Dict[int, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))

    with open(tracings_csv_path, newline="") as f:
        r = csv.DictReader(f)
        missing = TRACING_REQ - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"VolumeTracings.csv missing required columns: {sorted(missing)}")

        for row in r:
            fn = row["FileName"].strip()   # e.g. "XXXX.avi" (basename with ext)
            frame = int(row["Frame"])
            x1, y1 = float(row["X1"]), float(row["Y1"])
            tracings[fn][frame].append((x1, y1))  # FIX: use X1,Y1 only (in order)

    return tracings


# =========================
# Resampling helpers
# =========================

def resample_fps(video: torch.Tensor, src_fps: float, dst_fps: float, mode: str = "nearest") -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError("Expected video shape [T,C,H,W]")
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("FPS must be positive")

    T = int(video.size(0))
    if T == 0 or abs(src_fps - dst_fps) < 1e-6:
        return video

    new_T = max(1, int(round((T / src_fps) * dst_fps)))
    pos = torch.linspace(0, T - 1, new_T, device=video.device)

    if mode == "nearest":
        idx = pos.round().long().clamp_(0, T - 1)
        return video.index_select(0, idx)

    if mode == "linear":
        li = pos.floor().long().clamp_(0, T - 1)
        ri = (li + 1).clamp_(0, T - 1)
        a = (pos - li).view(-1, 1, 1, 1)
        return (1 - a) * video[li] + a * video[ri]

    raise ValueError("mode must be 'nearest' or 'linear'")

def resample_resolution(video: torch.Tensor, target=(128, 128), mode="bilinear") -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError("Expected video shape [T,C,H,W]")

    th, tw = (target, target) if isinstance(target, int) else target
    if th <= 0 or tw <= 0:
        raise ValueError("target must be positive")
    if tuple(video.shape[-2:]) == (th, tw):
        return video

    kwargs = {"size": (th, tw), "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(video, **kwargs)


# =========================
# Mask helpers (polygon -> binary)
# =========================

def map_old_frame_to_new_nearest(i_old: int, T_old: int, T_new: int) -> int:
    # Matches the index sampling from resample_fps(..., mode="nearest") for uniform linspace resampling.
    if T_new <= 0:
        raise ValueError("T_new must be positive")
    if T_old <= 1:
        return 0
    j = int(round(i_old * (T_new - 1) / (T_old - 1)))
    return max(0, min(T_new - 1, j))

def _clip_int_xy(x: float, y: float, W: int, H: int) -> Tuple[int, int]:
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0: xi = 0
    if yi < 0: yi = 0
    if xi > W - 1: xi = W - 1
    if yi > H - 1: yi = H - 1
    return xi, yi

def rasterize_polygon_pil(points_xy, out_hw):
    """
    Filled polygon mask (LV segmentation): returns uint8 mask in {0,1} of shape [H,W].
    """
    H, W = out_hw
    img = Image.new("L", (W, H), 0)
    if len(points_xy) >= 3:
        ImageDraw.Draw(img).polygon(points_xy, outline=1, fill=1)

    buf = bytearray(img.tobytes())                 # writable
    t = torch.frombuffer(buf, dtype=torch.uint8)   # no warning
    return (t.view(H, W) > 0).to(torch.uint8)

def build_lv_masks_for_video(
    *,
    tracings_for_file: Dict[int, List[Tuple[float, float]]],
    T_old: int,
    T_new: int,
    tracings_coord_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Builds per-frame LV filled masks for frames that have tracings.
    - tracings_for_file maps old frame index -> list of polygon vertices in tracings_coord_hw coordinates.
    - Frames are mapped into the resampled video using nearest-index alignment.

    Returns:
      {"frame_indices": LongTensor [N], "masks": UInt8Tensor [N,H,W]}
    """
    if not tracings_for_file:
        return None

    src_h, src_w = tracings_coord_hw
    dst_h, dst_w = dst_hw
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Bad tracings_coord_hw={tracings_coord_hw}")
    sx, sy = dst_w / float(src_w), dst_h / float(src_h)

    # If multiple tracings map to the same new frame (can happen after resampling),
    # keep the *last* one (typically fine).
    keep: Dict[int, torch.Tensor] = {}

    for i_old in sorted(tracings_for_file):
        pts = tracings_for_file[i_old]
        if len(pts) < 3:
            continue

        i_new = map_old_frame_to_new_nearest(i_old, T_old=T_old, T_new=T_new)

        # Scale, round, and clip to valid pixel coordinates (PIL tolerates floats, but clipping helps).
        pts_scaled = [_clip_int_xy(x * sx, y * sy, dst_w, dst_h) for (x, y) in pts]

        keep[i_new] = rasterize_polygon_pil(pts_scaled, (dst_h, dst_w))

    if not keep:
        return None

    frame_idx = sorted(keep.keys())
    masks = torch.stack([keep[i] for i in frame_idx], 0).to(torch.uint8)  # [N,H,W]

    return {
        "frame_indices": torch.tensor(frame_idx, dtype=torch.long),
        "masks": masks,
    }


# =========================
# Preprocess + saving
# =========================

def collate_fn(b):  # batch_size=1
    x = b[0]
    return x["video"], x["metadata"], x["filename"]

@torch.no_grad()
def preprocess_video(
    video: torch.Tensor, src_fps: float, dst_fps: float = 24.0, size=(128, 128),
    fps_mode: str = "nearest", resize_mode: str = "bilinear",
) -> torch.Tensor:
    return resample_resolution(resample_fps(video, src_fps, dst_fps, fps_mode), size, resize_mode)

def updated_metadata_after_preprocess(
    meta: Dict[str, Any], *, src_video: torch.Tensor, dst_video: torch.Tensor, dst_fps: float
) -> Dict[str, Any]:
    out = dict(meta)

    def keep_orig(k: str, v: Any):
        out.setdefault(f"Original{k}", v)

    keep_orig("FPS", float(out.get("FPS", dst_fps)))
    keep_orig("NumberOfFrames", int(out.get("NumberOfFrames", int(src_video.shape[0]))))
    keep_orig("FrameHeight", int(out.get("FrameHeight", int(src_video.shape[-2]))))
    keep_orig("FrameWidth", int(out.get("FrameWidth", int(src_video.shape[-1]))))

    out.update(
        FPS=float(dst_fps),
        NumberOfFrames=int(dst_video.shape[0]),
        FrameHeight=int(dst_video.shape[-2]),
        FrameWidth=int(dst_video.shape[-1]),
    )
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
    tracings: Optional[Dict[str, Dict[int, List[Tuple[float, float]]]]] = None,
    tracings_coord_hw: Optional[Tuple[int, int]] = None,
    masks_out_root: Optional[str] = None,
    save_masks: bool = True,
):
    split = split_name.upper()
    split_dir = os.path.join(out_root, split)
    os.makedirs(split_dir, exist_ok=True)

    masks_out_root = masks_out_root or (out_root + "_masks")
    masks_split_dir = os.path.join(masks_out_root, split)
    if save_masks:
        os.makedirs(masks_split_dir, exist_ok=True)

    n_ok = n_skip = n_fail = 0
    n_m_ok = n_m_skip = n_m_fail = 0

    for video, meta, filename in tqdm(dl, desc=f"Preprocessing {split}", unit="video"):
        meta = dict(meta)

        base = os.path.splitext(os.path.basename(filename))[0]
        out_path = os.path.join(split_dir, base + save_ext)

        processed_v = None  # keep in-memory result to avoid reload for masks

        try:
            if (not overwrite) and os.path.isfile(out_path):
                n_skip += 1
            else:
                src_fps = float(meta.get("FPS", 0.0))
                if src_fps <= 0:
                    raise ValueError(f"Bad src FPS in metadata: {meta.get('FPS')}")

                v = preprocess_video(video, src_fps, dst_fps, size, fps_mode, resize_mode)  # [T',C,h,w]
                meta_upd = updated_metadata_after_preprocess(meta, src_video=video, dst_video=v, dst_fps=dst_fps)

                v_save = v.permute(1, 0, 2, 3).contiguous().to(dtype=dtype).cpu()  # [C,T,h,w]
                processed_v = v_save

                payload = {
                    "video": v_save,
                    "fps": float(dst_fps),
                    "size": tuple(size) if not isinstance(size, int) else (int(size), int(size)),
                    "source_path": filename,
                    "metadata": meta_upd,
                }

                tmp = out_path + ".tmp"
                torch.save(payload, tmp)
                os.replace(tmp, out_path)
                n_ok += 1

        except Exception as e:
            n_fail += 1
            print(f"\n[{split}] FAILED (video): {filename}\n  -> {e}")
            continue

        # =========================
        # Masks (LV filled polygons)
        # =========================
        if not (save_masks and tracings is not None):
            continue

        try:
            mask_out_path = os.path.join(masks_split_dir, base + save_ext)
            if (not overwrite) and os.path.isfile(mask_out_path):
                n_m_skip += 1
                continue

            if fps_mode != "nearest":
                raise ValueError("Mask frame alignment currently assumes fps_mode='nearest'.")

            tracing_key = base + ".avi"
            if tracing_key not in tracings:
                n_m_skip += 1
                continue

            T_old = int(video.shape[0])

            # Determine destination H,W from preprocessed video.
            if processed_v is not None:
                T_new, h, w = int(processed_v.shape[1]), int(processed_v.shape[2]), int(processed_v.shape[3])
            else:
                vp = torch.load(out_path, map_location="cpu")
                T_new, h, w = int(vp["video"].shape[1]), int(vp["video"].shape[2]), int(vp["video"].shape[3])

            masks_pack = build_lv_masks_for_video(
                tracings_for_file=tracings[tracing_key],
                T_old=T_old,
                T_new=T_new,
                tracings_coord_hw=tracings_coord_hw,
                dst_hw=(h, w),
            )
            if masks_pack is None:
                n_m_skip += 1
                continue

            masks_payload = {
                "frame_indices": masks_pack["frame_indices"],  # [N]
                "masks": masks_pack["masks"],                  # [N,h,w] uint8 {0,1}
                "source_video_pt": out_path,
                "source_tracing_file": tracing_key,
                "dst_hw": (h, w),
                "tracings_coord_hw": tracings_coord_hw,
            }

            tmpm = mask_out_path + ".tmp"
            torch.save(masks_payload, tmpm)
            os.replace(tmpm, mask_out_path)
            n_m_ok += 1

        except Exception as e:
            n_m_fail += 1
            print(f"\n[{split}] FAILED (mask): {filename}\n  -> {e}")

    print(f"[{split}] videos: ok={n_ok}, skipped={n_skip}, failed={n_fail}")
    if save_masks:
        print(f"[{split}] masks : ok={n_m_ok}, skipped={n_m_skip}, failed={n_m_fail}")


# =========================
# Usage
# =========================

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(
        "data/echodyna/FileList.csv", "data/echodyna/Videos"
    )
    tracings = load_volume_tracings("data/echodyna/VolumeTracings.csv")

    workers = 16
    mkdl = lambda ds: DataLoader(ds, batch_size=1, num_workers=workers, collate_fn=collate_fn)

    out_root = "data/echodyna/preprocessed"
    masks_root = "data/echodyna/preprocessed_masks"

    for name, dl in [("TRAIN", mkdl(train_ds)), ("VAL", mkdl(val_ds)), ("TEST", mkdl(test_ds))]:
        save_preprocessed_split(
            dl, name,
            out_root=out_root, dst_fps=24.0, size=(128, 128),
            tracings=tracings,
            tracings_coord_hw=(112, 112),
            masks_out_root=masks_root, save_masks=True,
        )