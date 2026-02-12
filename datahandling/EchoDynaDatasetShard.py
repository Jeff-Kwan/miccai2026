import os, glob, io, tarfile
from typing import Dict, Any, Optional, List, Tuple

import torch
from torch.utils.data import Dataset


# Per-worker cache: {shard_path: tarfile.TarFile}
_TAR_CACHE: Dict[str, tarfile.TarFile] = {}


def _get_tar(shard_path: str) -> tarfile.TarFile:
    tf = _TAR_CACHE.get(shard_path)
    if tf is None:
        # r: (uncompressed) enables seeking
        tf = tarfile.open(shard_path, mode="r")
        _TAR_CACHE[shard_path] = tf
    return tf


class EchoDynaVideoShardDataset(Dataset):
    """
    Reads samples from tar shards.
    Each member is <base>.pt containing torch.save(payload) where payload contains:
      - video: uint8 [T,C,H,W]
      - fps, size, source_path, metadata
      - optional: masks {...}
    """
    def __init__(
        self,
        root: str,                          # e.g. data/echodyna/echoshards
        split: str,
        device: Optional[torch.device] = None,      # keep None to load on CPU
        video_dtype: Optional[torch.dtype] = None,  # upcast after normalize if you want
        return_masks: bool = True,
        weights_only: bool = True,
    ):
        self.split_dir = os.path.join(root, split.upper())
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Missing split dir: {self.split_dir}")

        self.shards = sorted(glob.glob(os.path.join(self.split_dir, "shard-*.tar")))
        if not self.shards:
            raise FileNotFoundError(f"No shard-*.tar found in: {self.split_dir}")

        self.device = device
        self.video_dtype = video_dtype
        self.return_masks = return_masks
        self.weights_only = weights_only

        # Build member index once (cheap compared to per-sample filesystem opens)
        self.index: List[Tuple[str, str]] = []
        for shard_path in self.shards:
            with tarfile.open(shard_path, mode="r") as tf:
                for m in tf.getmembers():
                    # only accept .pt samples
                    if m.isfile() and m.name.endswith(".pt") and "/" not in m.name:
                        self.index.append((shard_path, m.name))

        if not self.index:
            raise RuntimeError(f"Found shards but no .pt members in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_path, member_name = self.index[idx]

        tf = _get_tar(shard_path)
        fobj = tf.extractfile(member_name)
        if fobj is None:
            raise FileNotFoundError(f"Missing member {member_name} in {shard_path}")

        data = fobj.read()
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=self.weights_only)

        video = payload["video"]  # uint8 [T,C,H,W]

        # Normalize (keep here for compatibility; fastest is to do in collate/on-GPU)
        video = video.float().div_(255.0)
        if self.video_dtype is not None:
            video = video.to(self.video_dtype)
        if self.device is not None:
            video = video.to(self.device, non_blocking=True)

        fps = float(payload.get("fps", payload.get("metadata", {}).get("FPS", 0.0)))
        timestamps = torch.arange(video.shape[0], dtype=torch.float32) / max(fps, 1e-8)

        out = {
            "video": video,
            "timestamps": timestamps,
            "metadata": payload.get("metadata", {}),
            "fps": fps,
            "size": payload.get("size", None),
            "source_path": payload.get("source_path", None),
            "shard": shard_path,
            "key": member_name[:-3],  # base
        }

        if self.return_masks and "masks" in payload:
            out["masks"] = payload["masks"]

        return out


def load_echonet_dynamic_datasets(get_mask=False, root="data/echodyna/echoshards"):
    train_ds = EchoDynaVideoShardDataset(root, "TRAIN", video_dtype=torch.float32, return_masks=get_mask)
    val_ds   = EchoDynaVideoShardDataset(root, "VAL",   video_dtype=torch.float32, return_masks=get_mask)
    test_ds  = EchoDynaVideoShardDataset(root, "TEST",  video_dtype=torch.float32, return_masks=get_mask)
    return train_ds, val_ds, test_ds
