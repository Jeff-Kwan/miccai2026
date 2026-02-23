import os, io, csv, tarfile, warnings, multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
from torchvision.io import read_video
from PIL import Image, ImageDraw


# =========================
# CSV loading
# =========================

REQ = {"FileName", "EF", "ESV", "EDV", "FrameHeight", "FrameWidth", "FPS", "NumberOfFrames", "Split"}
META_CAST = {
    "EF": float, "ESV": float, "EDV": float,
    "FrameHeight": int, "FrameWidth": int,
    "FPS": float, "NumberOfFrames": int,
}

@dataclass(frozen=True)
class EchoDynaItem:
    video_path: str
    base: str
    split: str
    metadata: Dict[str, Any]

def load_split_items(csv_path: str, videos_dir: str) -> Dict[str, List[EchoDynaItem]]:
    splits: Dict[str, List[EchoDynaItem]] = {"TRAIN": [], "VAL": [], "TEST": []}
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            split = row["Split"].strip().upper()
            base = row["FileName"].strip()
            video_path = os.path.join(videos_dir, base + ".avi")
            meta = {k: META_CAST[k](row[k]) for k in META_CAST}
            splits[split].append(EchoDynaItem(video_path=video_path, base=base, split=split, metadata=meta))
    return splits

def load_volume_tracings(tracings_csv_path: str) -> Dict[str, Dict[int, List[Tuple[float, float]]]]:
    tracings: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
    with open(tracings_csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row["FileName"].strip()  # "XXXX.avi"
            frame = int(row["Frame"])
            pt = (float(row["X1"]), float(row["Y1"]))
            d = tracings.setdefault(fn, {})
            d.setdefault(frame, []).append(pt)
    return tracings


# =========================
# Masks
# =========================

def rasterize_polygon(points_xy: List[Tuple[int, int]], hw: Tuple[int, int]) -> torch.Tensor:
    H, W = hw
    img = Image.new("L", (W, H), 0)
    if len(points_xy) >= 3:
        ImageDraw.Draw(img).polygon(points_xy, outline=1, fill=1)
    buf = bytearray(img.tobytes())
    return (torch.frombuffer(buf, dtype=torch.uint8).view(H, W) > 0).to(torch.uint8)

def build_lv_masks(
    tracings_for_file: Dict[int, List[Tuple[float, float]]],
    T: int,
    tracings_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> Optional[Dict[str, torch.Tensor]]:
    if not tracings_for_file:
        return None

    src_h, src_w = tracings_hw
    dst_h, dst_w = dst_hw
    sx, sy = dst_w / src_w, dst_h / src_h

    keep: Dict[int, torch.Tensor] = {}
    for i in sorted(tracings_for_file):
        if i < 0 or i >= T:
            continue
        pts = tracings_for_file[i]
        if len(pts) < 3:
            continue

        pts_scaled = []
        for x, y in pts:
            xi = max(0, min(dst_w - 1, int(round(x * sx))))
            yi = max(0, min(dst_h - 1, int(round(y * sy))))
            pts_scaled.append((xi, yi))

        keep[i] = rasterize_polygon(pts_scaled, (dst_h, dst_w))

    if not keep:
        return None

    frame_idx = sorted(keep.keys())
    masks = torch.stack([keep[i] for i in frame_idx], 0).to(torch.uint8)  # [N,H,W]
    return {"frame_indices": torch.tensor(frame_idx, dtype=torch.long), "masks": masks}


# =========================
# Multiprocessing worker
# =========================

_G = {}

def _init_worker(
    tracings: Optional[Dict[str, Dict[int, List[Tuple[float, float]]]]],
    tracings_hw: Tuple[int, int],
    overwrite: bool,
    video_backend: Optional[str],
):
    torch.set_num_threads(1)
    if video_backend:
        try:
            from torchvision import set_video_backend
            set_video_backend(video_backend)
        except Exception:
            pass
    _G["tracings"] = tracings
    _G["tracings_hw"] = tracings_hw
    _G["overwrite"] = overwrite

def _process_one_to_bytes(item: EchoDynaItem) -> Tuple[str, str, Optional[bytes], bool, bool]:
    """
    Returns: (split, base, payload_bytes_or_None, video_ok, label_ok)
    If payload_bytes is None, caller can treat as "skipped" or "failed".
    label_ok means we successfully derived ED/ES.
    """
    tracings = _G["tracings"]
    tracings_hw = _G["tracings_hw"]

    split = item.split
    base = item.base
    video_path = item.video_path

    # ---- SKIP if no corresponding labels in VolumeTracings ----
    if tracings is None:
        return (split, base, None, False, False)
    tracing_key = base + ".avi"
    if tracing_key not in tracings:
        return (split, base, None, False, False)

    # Decode video
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
            )
            v, _, _ = read_video(video_path, pts_unit="sec")  # [T,H,W,C]
    except Exception:
        return (split, base, None, False, False)

    v_u8 = v.permute(0, 3, 1, 2).contiguous().to(torch.uint8).cpu()  # [T,C,H,W]
    T, C, H, W = v_u8.shape

    # Build masks (your implementation is already validated)
    masks_pack = build_lv_masks(tracings[tracing_key], T=T, tracings_hw=tracings_hw, dst_hw=(H, W))
    if masks_pack is None:
        return (split, base, None, True, False)

    frame_idx = masks_pack["frame_indices"]  # [N]
    masks = masks_pack["masks"]              # [N,H,W]

    # We assume there are exactly 2 indices (ED/ES) as per your note
    if frame_idx.numel() != 2 or masks.shape[0] != 2:
        return (split, base, None, True, False)

    # Compute areas and assign ED/ES by area
    areas = masks.view(2, -1).sum(dim=1).to(torch.long)  # [2]
    # ED = larger area, ES = smaller area
    if areas[0].item() >= areas[1].item():
        ed = int(frame_idx[0].item())
        es = int(frame_idx[1].item())
    else:
        ed = int(frame_idx[1].item())
        es = int(frame_idx[0].item())

    meta = dict(item.metadata)
    meta.update(
        FPS=float(meta.get("FPS", 0.0)),
        NumberOfFrames=int(T),
        FrameHeight=int(H),
        FrameWidth=int(W),
    )

    payload: Dict[str, Any] = {
        "video": v_u8,
        "fps": float(meta["FPS"]),
        "ED": ed,
        "ES": es,
        "metadata": meta,
    }

    bio = io.BytesIO()
    torch.save(payload, bio)
    return (split, base, bio.getvalue(), True, True)


# =========================
# Shard writer
# =========================

class TarShardWriter:
    """
    Writes uncompressed tar shards:
      out_root/SPLIT/shard-00000.tar, shard-00001.tar, ...
    Each sample is stored as: <base>.pt (bytes from torch.save)
    """
    def __init__(self, out_root: str, split: str, shard_size: int):
        self.split = split
        self.split_dir = os.path.join(out_root, split)
        os.makedirs(self.split_dir, exist_ok=True)
        self.shard_size = int(shard_size)

        self.shard_idx = 0
        self.count_in_shard = 0
        self.tar: Optional[tarfile.TarFile] = None
        self._open_new()

    def _open_new(self):
        if self.tar is not None:
            self.tar.close()
        path = os.path.join(self.split_dir, f"shard-{self.shard_idx:05d}.tar")
        self.tar = tarfile.open(path, mode="w")
        self.count_in_shard = 0
        self.shard_idx += 1

    def add_bytes(self, base: str, payload_bytes: bytes):
        assert self.tar is not None
        name = f"{base}.pt"
        ti = tarfile.TarInfo(name=name)
        ti.size = len(payload_bytes)
        self.tar.addfile(ti, io.BytesIO(payload_bytes))
        self.count_in_shard += 1
        if self.count_in_shard >= self.shard_size:
            self._open_new()

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


# =========================
# Run
# =========================

def preprocess_all_sharded(
    filelist_csv: str,
    videos_dir: str,
    tracings_csv: Optional[str],
    out_root: str,
    tracings_hw: Tuple[int, int] = (112, 112),
    overwrite: bool = True,
    video_backend: Optional[str] = None,
    processes: Optional[int] = None,
    chunksize: int = 4,
    shard_size: int = 256,
):
    splits = load_split_items(filelist_csv, videos_dir)
    tracings = load_volume_tracings(tracings_csv) if tracings_csv else None

    # Only keep tasks that have labels in VolumeTracings (skip the rest)
    if tracings is not None:
        def has_label(it: EchoDynaItem) -> bool:
            return (it.base + ".avi") in tracings
        tasks: List[EchoDynaItem] = [*filter(has_label, splits["TRAIN"] + splits["VAL"] + splits["TEST"])]
    else:
        tasks = []

    # If overwrite, clear existing split dirs
    if overwrite and os.path.isdir(out_root):
        for sp in ("TRAIN", "VAL", "TEST"):
            d = os.path.join(out_root, sp)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.endswith(".tar"):
                        try:
                            os.remove(os.path.join(d, fn))
                        except OSError:
                            pass

    writers = {
        "TRAIN": TarShardWriter(out_root, "TRAIN", shard_size),
        "VAL":   TarShardWriter(out_root, "VAL", shard_size),
        "TEST":  TarShardWriter(out_root, "TEST", shard_size),
    }

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=processes or os.cpu_count(),
        initializer=_init_worker,
        initargs=(tracings, tracings_hw, overwrite, video_backend),
    ) as pool:
        it = pool.imap_unordered(_process_one_to_bytes, tasks, chunksize=chunksize)

        v_ok = v_fail = lbl_ok = lbl_fail = 0
        for split, base, payload_bytes, video_ok, label_ok in tqdm(
            it, total=len(tasks), desc="Preprocessing(sharded)", unit="video"
        ):
            if payload_bytes is None:
                v_fail += int(not video_ok)
                lbl_fail += int(not label_ok)
                continue

            writers[split].add_bytes(base, payload_bytes)
            v_ok += int(video_ok)
            lbl_ok += int(label_ok)

    for w in writers.values():
        w.close()

    print(f"videos ok={v_ok} fail={v_fail}")
    print(f"labels ok={lbl_ok} fail/skip={lbl_fail}")


if __name__ == "__main__":
    preprocess_all_sharded(
        filelist_csv="data/echodyna/FileList.csv",
        videos_dir="data/echodyna/Videos",
        tracings_csv="data/echodyna/VolumeTracings.csv",
        out_root="data/echodyna/echoshards",
        tracings_hw=(112, 112),
        overwrite=True,
        video_backend=None,
        processes=48,
        chunksize=4,
        shard_size=256,
    )