import csv
import os
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
from torch.utils.data import Dataset
from torchvision import set_video_backend
from torchvision.io import read_video
from torchvision.transforms import Resize
from PIL import Image, ImageDraw


class EchoDynaDataset(Dataset):
    """
    A PyTorch Dataset for EchoNet-Dynamic videos.
    Returns:
    {"filename": str, "metadata": dict, "tracing": {"masks": List[Tensor], "frames": List[int]}}

    Args:
        entries: list of dicts with keys:
            - filename: str (full path to .avi)
            - metadata: dict (EF, ESV, EDV, FrameHeight, FrameWidth, FPS, NumberOfFrames)
            - (optional) tracings: list of dicts [{"frame": int, "points": List[Tuple[float,float]]}, ...]
        load_video: if True, loads the video using torchvision.io.read_video
        video_backend: optional; passed to torchvision.set_video_backend if available ("pyav" or "video_reader")
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        load_video: bool = False,
        video_backend: Optional[str] = None,
    ):
        self.entries = entries
        self.load_video = load_video
        self.video_backend = video_backend

        if self.load_video:
            try:
                if self.video_backend is not None:
                    set_video_backend(self.video_backend)
            except Exception as e:
                raise ImportError(
                    "load_video=True requires torchvision with video support. "
                    "Install torchvision (and PyAV if needed), or set load_video=False."
                ) from e

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _points_to_mask(
        points: List[Tuple[float, float]],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Convert a list of (x,y) polygon points into a filled binary mask [H, W] uint8.
        """
        if not points:
            return torch.zeros((height, width), dtype=torch.uint8)

        # Clamp + round to pixel coordinates
        poly = []
        for x, y in points:
            xi = int(round(x))
            yi = int(round(y))
            if xi < 0:
                xi = 0
            elif xi >= width:
                xi = width - 1
            if yi < 0:
                yi = 0
            elif yi >= height:
                yi = height - 1
            poly.append((xi, yi))

        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        # Fill polygon (standard EchoNet contour->mask convention)
        draw.polygon(poly, outline=1, fill=1)
        mask = torch.from_numpy(torch.ByteTensor(list(img.get_flattened_data())).view(height, width).numpy())
        return mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.entries[idx]
        out = {
            "filename": item["filename"],
            "metadata": item["metadata"],
        }

        if self.load_video:
            # read_video returns:
            #   video: Tensor[T, H, W, C] uint8
            #   audio: Tensor[...] (may be empty) or None depending on version/backend
            #   info: dict (contains 'video_fps' in many versions)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
                )
                video, _, _ = read_video(item["filename"], pts_unit="sec")

        # Video normalization to [-1, 1]
        video = video.float() / 127.5 - 1.0
        out["video"] = video.permute(3, 0, 1, 2)  # to [C, T, H, W]

        # Construct polygon mask from tracings
        tracings = item.get("tracings", [])
        H = int(item["metadata"].get("FrameHeight", 0))
        W = int(item["metadata"].get("FrameWidth", 0))

        out["tracing"] = {"EDES": [], "masks": [], "frames": []}
        for tr in tracings:
            frame = int(tr["frame"])
            pts = tr["points"]
            mask = self._points_to_mask(pts, H, W).unsqueeze(0)  # [1, H, W]
            # out["tracing"]["EDES"].append(out["video"][:, frame, :, :])
            out["tracing"]["masks"].append(mask)
            out["tracing"]["frames"].append(frame)
        # out["tracing"]["EDES"] = torch.stack(out["tracing"]["EDES"])
        # out["tracing"]["masks"] = torch.stack(out["tracing"]["masks"])

        # Resize video and masks so that they are 128x128
        if H != 128 or W != 128:
            resize_transform = Resize((128, 128))
            out["video"] = resize_transform(out["video"])
            resized_masks = []
            for mask in out["tracing"]["masks"]:
                resized_mask = resize_transform(mask.float()).byte()
                resized_masks.append(resized_mask)
            out["tracing"]["masks"] = resized_masks
        return out


def _parse_row_to_entry(row: Dict[str, str], videos_dir: str) -> Dict[str, Any]:
    filename = os.path.join(videos_dir, row["FileName"] + ".avi")

    metadata = {
        "EF": float(row["EF"]),
        "ESV": float(row["ESV"]),
        "EDV": float(row["EDV"]),
        "FrameHeight": int(row["FrameHeight"]),
        "FrameWidth": int(row["FrameWidth"]),
        "FPS": float(row["FPS"]),
        "NumberOfFrames": int(row["NumberOfFrames"]),
    }

    return {"filename": filename, "metadata": metadata, "FileName": row["FileName"]}


# --- NEW: parse VolumeTracings.csv into per-video, per-frame polygon points ---
def _load_volume_tracings(tracings_csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads VolumeTracings.csv with headers: FileName,X1,Y1,X2,Y2,Frame
    and returns:
        { FileName: [ {"frame": int, "points": [(x,y), ...]}, ... ] }
    Assumes rows for a given (FileName, Frame) are ordered along the contour.
    """
    by_key: Dict[Tuple[str, int], List[Tuple[float, float, float, float]]] = {}

    with open(tracings_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"FileName", "X1", "Y1", "X2", "Y2", "Frame"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"VolumeTracings CSV missing required columns: {sorted(missing)}")

        for row in reader:
            fn = row["FileName"].removesuffix(".avi")
            frame = int(float(row["Frame"]))
            x1 = float(row["X1"])
            y1 = float(row["Y1"])
            x2 = float(row["X2"])
            y2 = float(row["Y2"])
            by_key.setdefault((fn, frame), []).append((x1, y1, x2, y2))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for (fn, frame), segs in by_key.items():
        # Build polygon points from segment list:
        # take all X1,Y1 in order + last segment's X2,Y2 (EchoNet common convention)
        points: List[Tuple[float, float]] = [(s[0], s[1]) for s in segs]
        if segs:
            points.append((segs[-1][2], segs[-1][3]))

        out.setdefault(fn, []).append({"frame": frame, "points": points})
    return out


def load_echonet_dynamic_datasets(
    csv_path: str,
    videos_dir: str,
    tracings_dir: str,
    load_video: bool = False,
    video_backend: Optional[str] = None,
    verify_files_exist: bool = False,
) -> Tuple[EchoDynaDataset, EchoDynaDataset, EchoDynaDataset]:
    """
    Reads FileList.csv and returns (train_ds, val_ds, test_ds).

    Args:
        csv_path: path to FileList.csv
        videos_dir: path to Videos/ directory
        tracings_dir: path to VolumeTracings.csv
        load_video: if True, dataset loads video tensors via torchvision in __getitem__
        video_backend: optional backend string for torchvision video I/O ("pyav" or "video_reader")
        verify_files_exist: if True, raises FileNotFoundError if any video path doesn't exist
    """
    train_entries: List[Dict[str, Any]] = []
    val_entries: List[Dict[str, Any]] = []
    test_entries: List[Dict[str, Any]] = []

    # --- NEW: load tracings once and attach to entries ---
    tracings_map = _load_volume_tracings(tracings_dir)
    all_tracings = list(tracings_map.keys())

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
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
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            entry = _parse_row_to_entry(row, videos_dir)

            if verify_files_exist and not os.path.isfile(entry["filename"]):
                raise FileNotFoundError(f"Missing video file: {entry['filename']}")

            # --- NEW: attach list of tracings for this file (typically two frames) ---
            if entry["FileName"] in tracings_map:
                entry["tracings"] = tracings_map[entry["FileName"]]
            else:
                continue  # skip entries with no tracings

            split = row["Split"].strip().upper()
            if split == "TRAIN":
                train_entries.append(entry)
            elif split == "VAL":
                val_entries.append(entry)
            elif split == "TEST":
                test_entries.append(entry)
            else:
                raise ValueError(f"Unknown split value: {row['Split']}")

    train_ds = EchoDynaDataset(train_entries, load_video=load_video, video_backend=video_backend)
    val_ds = EchoDynaDataset(val_entries, load_video=load_video, video_backend=video_backend)
    test_ds = EchoDynaDataset(test_entries, load_video=load_video, video_backend=video_backend)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Minimal smoke test (adjust paths as needed):
    csv_path = "data/echodyna/FileList.csv"
    videos_dir = "data/echodyna/Videos"
    tracings_dir = "data/echodyna/VolumeTracings.csv"

    train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(
        csv_path,
        videos_dir,
        tracings_dir,
        load_video=True,        # set True if you want actual video tensors
        video_backend=None,      # e.g. "pyav"
        verify_files_exist=True  # set True to fail fast if videos are missing
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    ex = train_ds[0]
    print("Example item:", ex["filename"], ex["metadata"], ex["video"].shape)
    print("Tracing: frames", ex["tracing"]["frames"], "masks shape", ex["tracing"]["masks"][0].shape)
    # print("Tracing: EDES shape", ex["tracing"]["EDES"].shape)

    # # Matplotlib the tracing frame original frames on the left and masks on the right
    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    # for i in range(2):
    #     frame_idx = ex["tracing"]["frames"][i]
    #     mask = ex["tracing"]["masks"][i].numpy()
    #     frame = ex["video"][:, frame_idx, :, :].numpy()

    #     axs[i, 0].imshow(frame)
    #     axs[i, 0].set_title(f"Frame {frame_idx}")
    #     axs[i, 0].axis("off")

    #     axs[i, 1].imshow(mask, cmap="gray")
    #     axs[i, 1].set_title(f"Mask {frame_idx}")
    #     axs[i, 1].axis("off")
    # plt.tight_layout()
    # plt.savefig("example_tracing_masks.png")
    # plt.close()