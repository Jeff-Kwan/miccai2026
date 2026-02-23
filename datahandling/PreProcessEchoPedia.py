import os, io, csv, tarfile, warnings, multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
from torchvision.io import read_video
from PIL import Image, ImageDraw


# =========================
# CSV loading (Pediatric)
# =========================

META_CAST_PEDS = {
    "EF": float,
    "Sex": str,
    "Age": float,
    "Weight": float,
    "Height": float,
    "Split": str,   # kept if present in CSV; we just don't use it for foldering
}

@dataclass(frozen=True)
class EchoPedsItem:
    video_path: str
    base: str            # without .avi
    view: str            # A4C or PSAX
    metadata: Dict[str, Any]


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
                fn = row["FileName"].strip()  # includes ".avi"
                frame = int(row["Frame"])
                x = float(row["X"])
                y = float(row["Y"])
                tracings.setdefault(fn, {}).setdefault(frame, []).append((x, y))
            except Exception:
                continue
    return tracings


def load_items_peds_all(
    filelist_csv: str,
    videos_dir: str,
    view: str,
    tracings: Dict[str, Dict[int, List[Tuple[float, float]]]],
) -> List[EchoPedsItem]:
    """
    Load all items (no split usage).
    Only include samples with tracings AND exactly 2 traced frames.
    """
    items: List[EchoPedsItem] = []
    with open(filelist_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            avi_name = row["FileName"].strip()
            base = avi_name.removesuffix(".avi")

            t = tracings.get(avi_name)
            if not t or len(t.keys()) != 2:
                continue

            video_path = os.path.join(videos_dir, avi_name)

            meta: Dict[str, Any] = {}
            for k, caster in META_CAST_PEDS.items():
                if k not in row:
                    continue
                v = str(row[k]).strip()
                if v == "":
                    continue
                try:
                    meta[k] = caster(v)
                except Exception:
                    meta[k] = v

            # Keep view label in metadata
            meta["View"] = view

            items.append(
                EchoPedsItem(
                    video_path=video_path,
                    base=base,
                    view=view,
                    metadata=meta,
                )
            )
    return items


# =========================
# Polygon raster + area
# =========================

def rasterize_polygon(points_xy: List[Tuple[int, int]], hw: Tuple[int, int]) -> torch.Tensor:
    H, W = hw
    img = Image.new("L", (W, H), 0)
    if len(points_xy) >= 3:
        ImageDraw.Draw(img).polygon(points_xy, outline=1, fill=1)
    buf = bytearray(img.tobytes())
    return (torch.frombuffer(buf, dtype=torch.uint8).view(H, W) > 0).to(torch.uint8)

def polygon_mask_area_from_points(
    pts: List[Tuple[float, float]],
    hw: Tuple[int, int],
) -> int:
    H, W = hw
    if len(pts) < 3:
        return 0

    pts_i: List[Tuple[int, int]] = []
    for x, y in pts:
        xi = max(0, min(W - 1, int(round(x))))
        yi = max(0, min(H - 1, int(round(y))))
        pts_i.append((xi, yi))

    m = rasterize_polygon(pts_i, (H, W))
    return int(m.sum().item())


# =========================
# Multiprocessing worker
# =========================

_G: Dict[str, Any] = {}

def _init_worker(tracings_by_view: Dict[str, Dict[str, Dict[int, List[Tuple[float, float]]]]], video_backend: Optional[str]):
    torch.set_num_threads(1)
    if video_backend:
        try:
            from torchvision import set_video_backend
            set_video_backend(video_backend)
        except Exception:
            pass
    _G["tracings_by_view"] = tracings_by_view

def _process_one_to_bytes(item: EchoPedsItem) -> Tuple[str, str, Optional[bytes], bool, bool]:
    """
    Returns: (view, base, payload_bytes_or_None, video_ok, labels_ok)
    labels_ok = computed ED/ES successfully.
    """
    tracings_by_view = _G["tracings_by_view"]
    view, base = item.view, item.base
    video_path = item.video_path
    key = base + ".avi"

    tview = tracings_by_view.get(view, {})
    tracings_for_file = tview.get(key)
    if not tracings_for_file or len(tracings_for_file) != 2:
        return (view, base, None, False, False)

    # Decode video
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
            )
            v, _, info = read_video(video_path, pts_unit="sec")  # [T,H,W,C]
    except Exception:
        return (view, base, None, False, True)

    v_u8 = v.permute(0, 3, 1, 2).contiguous().to(torch.uint8).cpu()  # [T,C,H,W]
    T, C, H, W = v_u8.shape

    frames = sorted(tracings_for_file.keys())
    if len(frames) != 2:
        return (view, base, None, True, False)

    f0, f1 = int(frames[0]), int(frames[1])
    if not (0 <= f0 < T and 0 <= f1 < T):
        return (view, base, None, True, False)

    # Compute filled-polygon areas (ED larger, ES smaller)
    a0 = polygon_mask_area_from_points(tracings_for_file[f0], (H, W))
    a1 = polygon_mask_area_from_points(tracings_for_file[f1], (H, W))
    if a0 <= 0 or a1 <= 0:
        return (view, base, None, True, False)

    ed, es = (f0, f1) if a0 >= a1 else (f1, f0)

    # FPS from info if available; else 0.0
    fps = None
    try:
        if isinstance(info, dict):
            fps = info.get("video_fps", None)
    except Exception:
        pass
    if fps is None:
        fps = item.metadata.get("FPS", 0.0)
    fps = float(fps) if fps is not None else 0.0

    payload: Dict[str, Any] = {
        "video": v_u8,
        "fps": fps,
        "ED": int(ed),
        "ES": int(es),
        "metadata": dict(item.metadata),
    }

    bio = io.BytesIO()
    torch.save(payload, bio)
    return (view, base, bio.getvalue(), True, True)


# =========================
# Shard writer (by view only)
# =========================

class TarShardWriter:
    def __init__(self, out_root: str, view: str, shard_size: int):
        self.view_dir = os.path.join(out_root, view)
        os.makedirs(self.view_dir, exist_ok=True)
        self.shard_size = int(shard_size)
        self.shard_idx = 0
        self.count_in_shard = 0
        self.tar: Optional[tarfile.TarFile] = None
        self._open_new()

    def _open_new(self):
        if self.tar is not None:
            self.tar.close()
        path = os.path.join(self.view_dir, f"shard-{self.shard_idx:05d}.tar")
        self.tar = tarfile.open(path, mode="w")  # uncompressed
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
# Run (whole dataset, per view)
# =========================

def preprocess_peds_sharded_all(
    data_root: str,
    out_root: str,
    views: Tuple[str, ...] = ("A4C", "PSAX"),
    overwrite: bool = True,
    video_backend: Optional[str] = None,
    processes: Optional[int] = None,
    chunksize: int = 4,
    shard_size: int = 256,
):
    # optional cleanup (remove shards under each view)
    if overwrite and os.path.isdir(out_root):
        for view in views:
            d = os.path.join(out_root, view)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.endswith(".tar"):
                        try:
                            os.remove(os.path.join(d, fn))
                        except OSError:
                            pass

    tracings_by_view: Dict[str, Dict[str, Dict[int, List[Tuple[float, float]]]]] = {}
    tasks: List[EchoPedsItem] = []

    for view in views:
        view_root = os.path.join(data_root, view)
        videos_dir = os.path.join(view_root, "Videos")
        filelist_csv = os.path.join(view_root, "FileList.csv")
        tracings_csv = os.path.join(view_root, "VolumeTracings.csv")

        tracings = load_tracings_polygons_xy(tracings_csv)
        tracings_by_view[view] = tracings

        items = load_items_peds_all(filelist_csv, videos_dir, view=view, tracings=tracings)
        tasks.extend(items)

    writers: Dict[str, TarShardWriter] = {view: TarShardWriter(out_root, view, shard_size) for view in views}

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=processes or os.cpu_count(),
        initializer=_init_worker,
        initargs=(tracings_by_view, video_backend),
    ) as pool:
        it = pool.imap_unordered(_process_one_to_bytes, tasks, chunksize=chunksize)

        v_ok = v_fail = lbl_ok = lbl_fail = 0
        for view, base, payload_bytes, video_ok, labels_ok in tqdm(
            it, total=len(tasks), desc="Preprocessing Peds (ED/ES via polygon area)", unit="video"
        ):
            if payload_bytes is None:
                v_fail += int(not video_ok)
                lbl_fail += int(not labels_ok)
                continue

            writers[view].add_bytes(base, payload_bytes)
            v_ok += int(video_ok)
            lbl_ok += int(labels_ok)

    for w in writers.values():
        w.close()

    print(f"videos ok={v_ok} fail={v_fail}")
    print(f"labels ok={lbl_ok} fail/skip={lbl_fail}")
    print(f"output: {out_root}/<A4C|PSAX>/shard-*.tar")


if __name__ == "__main__":
    preprocess_peds_sharded_all(
        data_root="./data/echonetpediatric",
        out_root="./data/echonetpediatric/echoshards",
        views=("A4C", "PSAX"),
        overwrite=True,
        video_backend=None,
        processes=32,
        chunksize=4,
        shard_size=256,
    )