import os
import io
import tarfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class EchoPediaShardDataset(Dataset):
    """
    Minimal dataset for shards written as:
      out_root/<view>/shard-*.tar
    where each tar member is "<base>.pt" created with torch.save({...}).

    Returns the loaded dict (or dict + key if return_key=True).
    """

    def __init__(
        self,
        shards_root: str,
        views: Union[str, List[str], Tuple[str, ...]] = ("A4C", "PSAX"),
        return_key: bool = False,
    ):
        self.shards_root = shards_root
        if isinstance(views, str):
            views = [views]
        self.views = list(views)
        self.return_key = return_key

        # Flat index: [(tar_path, member_name), ...]
        self.index: List[Tuple[str, str]] = []
        self._build_index()

        # Per-worker tar handles: {tar_path: tarfile.TarFile}
        self._tars: Dict[str, tarfile.TarFile] = {}

    def _build_index(self):
        for view in self.views:
            view_dir = os.path.join(self.shards_root, view)
            if not os.path.isdir(view_dir):
                continue
            tar_files = sorted(
                os.path.join(view_dir, fn)
                for fn in os.listdir(view_dir)
                if fn.endswith(".tar")
            )
            for tar_path in tar_files:
                # Read member list once (cheap) and store names
                with tarfile.open(tar_path, "r") as tf:
                    for m in tf.getmembers():
                        if m.isfile() and m.name.endswith(".pt"):
                            self.index.append((tar_path, m.name))

    def __len__(self) -> int:
        return len(self.index)

    def _get_tar(self, tar_path: str) -> tarfile.TarFile:
        # Lazily open per worker/process; keeps it fast during iteration.
        tf = self._tars.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path, "r")
            self._tars[tar_path] = tf
        return tf

    def __getitem__(self, idx: int):
        tar_path, member_name = self.index[idx]
        tf = self._get_tar(tar_path)

        f = tf.extractfile(member_name)
        if f is None:
            raise FileNotFoundError(f"Missing member {member_name} in {tar_path}")

        payload_bytes = f.read()
        sample = torch.load(io.BytesIO(payload_bytes), map_location="cpu")  # dict

        # Normalize video
        sample["video"] = sample["video"].float().div_(255.0)

        if self.return_key:
            # key like "A4C/shard-00001.tar:XXXX.pt"
            view = os.path.basename(os.path.dirname(tar_path))
            key = f"{view}/{os.path.basename(tar_path)}:{member_name}"
            return sample, key

        return sample

    def close(self):
        for tf in self._tars.values():
            try:
                tf.close()
            except Exception:
                pass
        self._tars.clear()

    def __del__(self):
        self.close()


def get_echopedia_shard_dataset(
    shards_root: str = "data/echonetpediatric/echoshards",
    views: Union[str, List[str], Tuple[str, ...]] = ("A4C", "PSAX"),
    return_key: bool = False,
) -> EchoPediaShardDataset:
    return EchoPediaShardDataset(
        shards_root=shards_root,
        views=views,
        return_key=return_key,
    )


if __name__ == "__main__":
    # Quick test
    ds = get_echopedia_shard_dataset(return_key=True)
    print(f"Dataset length: {len(ds)}")
    sample, key = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample key: {key}")
    print(sample['fps'])