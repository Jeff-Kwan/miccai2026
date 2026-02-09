import os, glob
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset

class EchoDynaVideoDataset(Dataset):
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
    train_ds = EchoDynaVideoDataset(pre_root, "TRAIN", video_dtype=torch.float32)
    val_ds   = EchoDynaVideoDataset(pre_root, "VAL",   video_dtype=torch.float32)
    test_ds  = EchoDynaVideoDataset(pre_root, "TEST",  video_dtype=torch.float32)
    return train_ds, val_ds, test_ds



class EchoDynaDownstreamDataset(Dataset):
    """
    Loads BOTH:
      1) preprocessed video payloads saved by save_preprocessed_split() into <video_root>/<SPLIT>/*.pt
      2) corresponding mask payloads saved by save_preprocessed_split() into <masks_root>/<SPLIT>/*.pt

    Video payload expected:
      {
        "video": [C,T,H,W] (float16/float32),
        "fps": float,
        "size": (H,W),
        "source_path": str,
        "metadata": dict,
      }

    Mask payload expected:
      {
        "frame_indices": [N] long,
        "masks": [N,H,W] uint8,
        "source_video_pt": str,
        "source_tracing_file": str,
        "dst_hw": (H,W),
        "tracing_coord_hw": (h0,w0),
      }

    Returns an item dict with everything:
      {
        "video": Tensor [C,T,H,W],
        "metadata": dict,
        "fps": float|None,
        "size": (H,W)|None,
        "source_path": str|None,

        "frame_indices": LongTensor [N] (or empty if missing),
        "masks": ByteTensor [N,H,W] (or empty if missing),
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

        self.video_split_dir = os.path.join(video_root, self.split)
        if not os.path.isdir(self.video_split_dir):
            raise FileNotFoundError(f"Missing video split dir: {self.video_split_dir}")

        self.masks_split_dir = os.path.join(masks_root, self.split)
        self.has_masks_dir = os.path.isdir(self.masks_split_dir)

        video_paths = sorted(glob.glob(os.path.join(self.video_split_dir, f"*{ext}")))
        if len(video_paths) == 0:
            raise FileNotFoundError(f"No {ext} files found in: {self.video_split_dir}")

        self.device = device
        self.video_dtype = video_dtype
        self.masks_device = masks_device if masks_device is not None else device
        self.allow_missing_masks = allow_missing_masks
        self.ext = ext

        # Build basename -> path maps
        self._video_by_base: Dict[str, str] = {
            os.path.splitext(os.path.basename(p))[0]: p for p in video_paths
        }

        self._mask_by_base: Dict[str, str] = {}
        if self.has_masks_dir:
            mask_paths = glob.glob(os.path.join(self.masks_split_dir, f"*{ext}"))
            self._mask_by_base = {
                os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths
            }

            # Keep ONLY paired items:
            # - skip videos missing masks
            # - skip masks missing videos
            common_bases = sorted(set(self._video_by_base).intersection(self._mask_by_base))
            self.video_paths = [self._video_by_base[b] for b in common_bases]
            # shrink mask map to paired only (optional but keeps everything consistent)
            self._mask_by_base = {b: self._mask_by_base[b] for b in common_bases}

            # If masks dir exists and pairing yields nothing, it's usually a config/data issue.
            if (not self.allow_missing_masks) and len(self.video_paths) == 0:
                raise FileNotFoundError(
                    f"No paired (video, mask) files found in: {self.video_split_dir} and {self.masks_split_dir}"
                )

        else:
            # No masks dir: keep original behavior
            if not self.allow_missing_masks:
                raise FileNotFoundError(f"Missing masks split dir: {self.masks_split_dir}")
            self.video_paths = video_paths

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        base = os.path.splitext(os.path.basename(video_path))[0]

        vp = torch.load(video_path, map_location="cpu")
        video = vp["video"]  # [C,T,H,W]

        if self.video_dtype is not None:
            video = video.to(self.video_dtype)
        if self.device is not None:
            video = video.to(self.device, non_blocking=True)

        # Default: no masks
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
            # In paired-only mode this shouldn't happen, but keep it as a guardrail.
            raise FileNotFoundError(f"Missing masks for item: {base} ({video_path})")

        return {
            "video": video,
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


# Example usage
def load_echodyna_downstream_datasets(allow_missing_masks: bool = True):
    pre_root = "data/echodyna/preprocessed"
    masks_root = "data/echodyna/preprocessed_masks"

    train_ds = EchoDynaDownstreamDataset(pre_root, masks_root, "TRAIN", video_dtype=torch.float32, allow_missing_masks=allow_missing_masks)
    val_ds   = EchoDynaDownstreamDataset(pre_root, masks_root, "VAL",   video_dtype=torch.float32, allow_missing_masks=allow_missing_masks)
    test_ds  = EchoDynaDownstreamDataset(pre_root, masks_root, "TEST",  video_dtype=torch.float32, allow_missing_masks=allow_missing_masks)
    return train_ds, val_ds, test_ds
