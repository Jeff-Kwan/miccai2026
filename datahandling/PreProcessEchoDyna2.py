import os, csv, warnings, multiprocessing as mp
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

            d = tracings.get(fn)
            if d is None:
                d = {}
                tracings[fn] = d

            lst = d.get(frame)
            if lst is None:
                lst = []
                d[frame] = lst

            lst.append(pt)

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
    out_root: str,
    masks_root: str,
    dtype_str: str,
    save_ext: str,
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

    dtype = getattr(torch, dtype_str)

    _G["out_root"] = out_root
    _G["masks_root"] = masks_root
    _G["dtype"] = dtype
    _G["save_ext"] = save_ext
    _G["tracings"] = tracings
    _G["tracings_hw"] = tracings_hw
    _G["overwrite"] = overwrite

def _process_one(item: EchoDynaItem) -> Tuple[str, bool, bool, str]:
    """
    Returns: (split, video_ok, mask_ok, base)
    """
    out_root = _G["out_root"]
    masks_root = _G["masks_root"]
    dtype = _G["dtype"]
    save_ext = _G["save_ext"]
    tracings = _G["tracings"]
    tracings_hw = _G["tracings_hw"]
    overwrite = _G["overwrite"]

    split = item.split
    base = item.base
    video_path = item.video_path

    split_dir = os.path.join(out_root, split)
    masks_split_dir = os.path.join(masks_root, split)
    os.makedirs(split_dir, exist_ok=True)
    if tracings is not None:
        os.makedirs(masks_split_dir, exist_ok=True)

    out_path = os.path.join(split_dir, base + save_ext)

    video_ok = False
    mask_ok = False

    # ---- decode + save video ----
    if overwrite or not os.path.isfile(out_path):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
            )
            v, _, _ = read_video(video_path, pts_unit="sec")  # [T,H,W,C]

        # Convert to [T,C,H,W] and normalize to [0,1]
        v_save = (
            v.permute(0, 3, 1, 2)   # [T,C,H,W]
            .contiguous()
            .to(torch.uint8)        # keep [0,255] uint8
            .cpu()
        )


        meta = dict(item.metadata)
        meta.update(
            FPS=float(meta.get("FPS", 0.0)),
            NumberOfFrames=int(v.shape[0]),
            FrameHeight=int(v.shape[-2]),
            FrameWidth=int(v.shape[-1]),
        )

        payload = {
            "video": v_save,
            "fps": float(meta["FPS"]),
            "size": (int(v_save.shape[2]), int(v_save.shape[3])),  # (H,W)
            "source_path": video_path,
            "metadata": meta,
        }

        tmp = out_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, out_path)
        video_ok = True
    else:
        video_ok = True

    # ---- masks ----
    if tracings is None:
        return (split, video_ok, mask_ok, base)

    tracing_key = base + ".avi"
    if tracing_key not in tracings:
        return (split, video_ok, mask_ok, base)

    mask_out_path = os.path.join(masks_split_dir, base + save_ext)
    if (not overwrite) and os.path.isfile(mask_out_path):
        return (split, video_ok, True, base)

    # If video was skipped, load minimal info from saved file
    if overwrite or video_ok:
        if overwrite or not os.path.isfile(out_path):
            return (split, video_ok, mask_ok, base)
    vp = torch.load(out_path, map_location="cpu")
    T, C, H, W = vp["video"].shape

    masks_pack = build_lv_masks(tracings[tracing_key], T=T, tracings_hw=tracings_hw, dst_hw=(H, W))
    if masks_pack is None:
        return (split, video_ok, mask_ok, base)

    masks_payload = {
        "frame_indices": masks_pack["frame_indices"],
        "masks": masks_pack["masks"],
        "source_video_pt": out_path,
        "source_tracing_file": tracing_key,
        "dst_hw": (H, W),
        "tracings_hw": tracings_hw,
    }

    tmpm = mask_out_path + ".tmp"
    torch.save(masks_payload, tmpm)
    os.replace(tmpm, mask_out_path)
    mask_ok = True

    return (split, video_ok, mask_ok, base)


# =========================
# Run
# =========================

def preprocess_all(
    filelist_csv: str,
    videos_dir: str,
    tracings_csv: Optional[str],
    out_root: str,
    masks_root: str,
    tracings_hw: Tuple[int, int] = (112, 112),
    overwrite: bool = False,
    dtype: torch.dtype = torch.float16,
    save_ext: str = ".pt",
    video_backend: Optional[str] = None,
    processes: Optional[int] = None,
    chunksize: int = 4,
):
    splits = load_split_items(filelist_csv, videos_dir)
    tracings = load_volume_tracings(tracings_csv) if tracings_csv else None

    tasks: List[EchoDynaItem] = splits["TRAIN"] + splits["VAL"] + splits["TEST"]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=processes or os.cpu_count(),
        initializer=_init_worker,
        initargs=(
            out_root,
            masks_root,
            str(dtype).replace("torch.", ""),
            save_ext,
            tracings,
            tracings_hw,
            overwrite,
            video_backend,
        ),
    ) as pool:
        it = pool.imap_unordered(_process_one, tasks, chunksize=chunksize)

        v_ok = v_fail = m_ok = m_fail = 0
        for split, video_ok, mask_ok, base in tqdm(it, total=len(tasks), desc="Preprocessing", unit="video"):
            v_ok += int(video_ok)
            v_fail += int(not video_ok)
            m_ok += int(mask_ok)
            m_fail += int(tracings is not None and not mask_ok)

    print(f"videos ok={v_ok} fail={v_fail}")
    if tracings is not None:
        print(f"masks  ok={m_ok} fail/skip={m_fail}")


if __name__ == "__main__":
    preprocess_all(
        filelist_csv="data/echodyna/FileList.csv",
        videos_dir="data/echodyna/Videos",
        tracings_csv="data/echodyna/VolumeTracings.csv",
        out_root="data/echodyna/echovids",
        masks_root="data/echodyna/echomasks",
        tracings_hw=(112, 112),
        overwrite=False,
        dtype=torch.float16,
        save_ext=".pt",
        video_backend=None,
        processes=64,
        chunksize=4,
    )