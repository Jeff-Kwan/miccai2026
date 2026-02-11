import os, glob
from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset


class EchoDynaVideoDataset(Dataset):
    """
    Expects each .pt payload:
      {
        "video": Tensor [T,C,H,W] (float16/float32), normalized to [0,1],
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
        device: Optional[torch.device] = None,          # keep None to load on CPU
        video_dtype: Optional[torch.dtype] = None,      # e.g. torch.float32 to upcast
    ):
        self.split_dir = os.path.join(root, split.upper())
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Missing split dir: {self.split_dir}")

        self.paths = sorted(glob.glob(os.path.join(self.split_dir, f"*{ext}")))
        if not self.paths:
            raise FileNotFoundError(f"No {ext} files found in: {self.split_dir}")

        self.device = device
        self.video_dtype = video_dtype

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        payload = torch.load(path, map_location="cpu")

        video = payload["video"]  # [T,C,H,W]
        video = video.float().div_(255.0)   # normalize to [0,1]
        if self.video_dtype is not None:
            video = video.to(self.video_dtype)
        if self.device is not None:
            video = video.to(self.device, non_blocking=True)

        return {
            "video": video,
            "timestamps": torch.arange(video.shape[0]) / payload["metadata"]["FPS"],
            "metadata": payload.get("metadata", {}),
            "fps": payload.get("fps", None),
            "size": payload.get("size", None),
            "source_path": payload.get("source_path", None),
            "path": path,
        }


def load_echonet_dynamic_datasets():
    pre_root = "data/echodyna/echovids"
    train_ds = EchoDynaVideoDataset(pre_root, "TRAIN", video_dtype=torch.float32)
    val_ds   = EchoDynaVideoDataset(pre_root, "VAL",   video_dtype=torch.float32)
    test_ds  = EchoDynaVideoDataset(pre_root, "TEST",  video_dtype=torch.float32)
    return train_ds, val_ds, test_ds


class EchoDynaDownstreamDataset(Dataset):
    """
    Loads video + masks.

    Video payload expected:
      {
        "video": Tensor [T,C,H,W] (float16/float32), normalized to [0,1],
        "fps": float,
        "size": (H,W),
        "source_path": str,
        "metadata": dict,
      }

    Mask payload expected:
      {
        "frame_indices": LongTensor [N],
        "masks": ByteTensor [N,H,W],
        "source_video_pt": str,
        "source_tracing_file": str,
        "dst_hw": (H,W),
        "tracings_hw": (h0,w0),
      }

    Returns:
      {
        "video": Tensor [T,C,H,W],
        "metadata": dict,
        "fps": float|None,
        "size": (H,W)|None,
        "source_path": str|None,

        "frame_indices": LongTensor [N] (or empty),
        "masks": ByteTensor [N,H,W] (or empty),
        "has_masks": bool,

        "video_path": str,
        "mask_path": str|None,
      }
    """
    def __init__(
        self,
        video_root: str,
        masks_root: str,
        split: str,
        ext: str = ".pt",
        device: Optional[torch.device] = None,
        video_dtype: Optional[torch.dtype] = None,
        masks_device: Optional[torch.device] = None,
        allow_missing_masks: bool = True,
    ):
        self.split = split.upper()
        self.ext = ext

        self.video_split_dir = os.path.join(video_root, self.split)
        if not os.path.isdir(self.video_split_dir):
            raise FileNotFoundError(f"Missing video split dir: {self.video_split_dir}")

        self.masks_split_dir = os.path.join(masks_root, self.split)
        self.has_masks_dir = os.path.isdir(self.masks_split_dir)

        video_paths = sorted(glob.glob(os.path.join(self.video_split_dir, f"*{ext}")))
        if not video_paths:
            raise FileNotFoundError(f"No {ext} files found in: {self.video_split_dir}")

        self.device = device
        self.video_dtype = video_dtype
        self.masks_device = masks_device if masks_device is not None else device
        self.allow_missing_masks = allow_missing_masks

        self._video_by_base = {os.path.splitext(os.path.basename(p))[0]: p for p in video_paths}

        self._mask_by_base = {}
        if self.has_masks_dir:
            mask_paths = glob.glob(os.path.join(self.masks_split_dir, f"*{ext}"))
            self._mask_by_base = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}

            common = sorted(set(self._video_by_base).intersection(self._mask_by_base))
            self.video_paths = [self._video_by_base[b] for b in common]
            self._mask_by_base = {b: self._mask_by_base[b] for b in common}

            if (not self.allow_missing_masks) and not self.video_paths:
                raise FileNotFoundError(
                    f"No paired (video, mask) files found in: {self.video_split_dir} and {self.masks_split_dir}"
                )
        else:
            if not self.allow_missing_masks:
                raise FileNotFoundError(f"Missing masks split dir: {self.masks_split_dir}")
            self.video_paths = video_paths

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        base = os.path.splitext(os.path.basename(video_path))[0]

        vp = torch.load(video_path, map_location="cpu")
        video = vp["video"]  # [T,C,H,W]
        video = video.float().div_(255.0)   # normalize to [0,1]

        if self.video_dtype is not None:
            video = video.to(self.video_dtype)
        if self.device is not None:
            video = video.to(self.device, non_blocking=True)

        frame_indices = torch.empty((0,), dtype=torch.long)
        masks = torch.empty((0, 0, 0), dtype=torch.uint8)
        mask_path = None
        has_masks = False

        if self.has_masks_dir and (base in self._mask_by_base):
            mask_path = self._mask_by_base[base]
            mp = torch.load(mask_path, map_location="cpu")

            frame_indices = mp.get("frame_indices", frame_indices)
            masks = mp.get("masks", masks)
            has_masks = True

            if self.masks_device is not None:
                frame_indices = frame_indices.to(self.masks_device, non_blocking=True)
                masks = masks.to(self.masks_device, non_blocking=True)

        elif (not self.allow_missing_masks) and self.has_masks_dir:
            raise FileNotFoundError(f"Missing masks for item: {base} ({video_path})")

        return {
            "video": video,
            "timestamps": torch.arange(video.shape[0]) / vp["metadata"]["FPS"],
            "metadata": vp.get("metadata", {}),
            "fps": vp.get("fps", None),
            "size": vp.get("size", None),
            "source_path": vp.get("source_path", None),

            "frame_indices": frame_indices,
            "masks": masks,
            "has_masks": has_masks,

            "video_path": video_path,
            "mask_path": mask_path,
        }


def load_echodyna_downstream_datasets(allow_missing_masks: bool = False):
    pre_root = "data/echodyna/echovids"
    masks_root = "data/echodyna/echomasks"

    train_ds = EchoDynaDownstreamDataset(
        pre_root, masks_root, "TRAIN", video_dtype=torch.float32, allow_missing_masks=allow_missing_masks
    )
    val_ds = EchoDynaDownstreamDataset(
        pre_root, masks_root, "VAL", video_dtype=torch.float32, allow_missing_masks=allow_missing_masks
    )
    test_ds = EchoDynaDownstreamDataset(
        pre_root, masks_root, "TEST", video_dtype=torch.float32, allow_missing_masks=allow_missing_masks
    )
    return train_ds, val_ds, test_ds
