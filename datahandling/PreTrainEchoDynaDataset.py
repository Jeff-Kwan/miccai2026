import os, glob
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

class EchoDynaPreprocessedDataset(Dataset):
    """
    Reads files saved by save_preprocessed_split():
      payload = {
        "video": [C,T,H,W] (float16/float32),
        "fps": float,
        "size": (H,W),
        "source_path": str,
        "metadata": dict,
      }
    """
    def __init__(
        self,
        root: str,
        split: str,
        ext: str = ".pt",
        device: Optional[torch.device] = None,   # keep None to load on CPU
        video_dtype: Optional[torch.dtype] = None,  # e.g. torch.float32 to upcast
    ):
        self.split_dir = os.path.join(root, split.upper())
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Missing split dir: {self.split_dir}")

        self.paths = sorted(glob.glob(os.path.join(self.split_dir, f"*{ext}")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No {ext} files found in: {self.split_dir}")

        self.device = device
        self.video_dtype = video_dtype

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        payload = torch.load(path, map_location="cpu")  # keep CPU; let DataLoader pin/copy

        video = payload["video"]  # [C,T,H,W]
        if self.video_dtype is not None:
            video = video.to(self.video_dtype)
        if self.device is not None:
            video = video.to(self.device, non_blocking=True)

        return {
            "video": video,
            "metadata": payload.get("metadata", {}),
            "fps": payload.get("fps", None),
            "size": payload.get("size", None),
            "source_path": payload.get("source_path", None),
            "path": path,
        }

def load_echonet_dynamic_datasets():
    pre_root = "data/echodyna/preprocessed"
    train_ds = EchoDynaPreprocessedDataset(pre_root, "TRAIN", video_dtype=torch.float32)
    val_ds   = EchoDynaPreprocessedDataset(pre_root, "VAL",   video_dtype=torch.float32)
    test_ds  = EchoDynaPreprocessedDataset(pre_root, "TEST",  video_dtype=torch.float32)
    return train_ds, val_ds, test_ds