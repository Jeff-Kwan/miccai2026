import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch.utils.data import Dataset


class EchoDynaLatentsDataset(Dataset):
    """
    Loads trajectory latent .pt files produced by your script:
      /data/echodyna/latents/{split_name}/{idx:04d}.pt

    Each file is expected to be a dict with keys:
      z, z_spline, timestamps, phase, frame_indices, fps, metadata
    """

    def __init__(
        self,
        root: Union[str, Path] = "/data/echodyna/latents",
        split: str = "Train",
        map_location: Union[str, torch.device, None] = "cpu",
        strict: bool = True,
    ):
        """
        Args:
            root: Base directory containing split folders.
            split: Split folder name ("Train", "Val", "Test").
            keys: If provided, only return these keys from each sample.
            map_location: Passed to torch.load (recommend "cpu" for dataloader).
            transform: Optional callable(sample_dict) -> sample_dict.
            strict: If True, error if directory missing or no .pt files found.
        """
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.map_location = map_location

        if strict and not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.files: List[Path] = sorted(self.split_dir.glob("*.pt"))

        if strict and len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        sample: Dict[str, Any] = torch.load(path, map_location=self.map_location, weights_only=False)

        # Add handy info
        sample["__path__"] = str(path)
        sample["__index__"] = idx
        return sample


def get_latents_dataset():
    train_ds = EchoDynaLatentsDataset(root="/data/echodyna/latents", split="Train")
    val_ds = EchoDynaLatentsDataset(root="/data/echodyna/latents", split="Val")
    test_ds = EchoDynaLatentsDataset(root="/data/echodyna/latents", split="Test")
    return train_ds, val_ds, test_ds


# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = EchoDynaLatentsDataset(
        root="/data/echodyna/latents",
        split="Train",
        map_location="cpu",
    )

    sample = next(iter(ds))
    print(sample.keys())
    print(sample["z"].shape)        # if tensors/arrays were saved as tensors
    print(sample["metadata"])