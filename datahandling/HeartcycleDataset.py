from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RecordMeta:
    """Metadata parsed from filename like CH07_59146237_s0000029.h5."""
    subject_id: Optional[str]
    experiment_id: Optional[str]
    record_id: Optional[str]
    path: str


class HeartcycleDataset(Dataset):
    """
    HeartCycle PyTorch Dataset.

    Reads per-record HDF5 files that store synchronized multi-modal signals.
    Signal group IDs and shapes follow the dataset README, e.g.:
      - _030 ECG (Niccomo), _031 IMP (Niccomo), _061 PCG (Stethoscope),
        _091 ECHO (Echo device), _121 PPG (PPG device), etc.
      - Most signals are stored as (1, time) and are squeezed to (time,).
      - AVO/AVC are event-time arrays; PEP/LVET are scalar or 1D durations (ms).
      - ECHO (_091) is stored as (3, time, depth/velocity) and (optionally)
        transposed to (3, depth/velocity, time) for image-like usage.

    Returns a dict by default:
      {
        "x": {<input_key>: tensor, ...},
        "t": {<key>: time_tensor, ...}   # if include_time=True
        "rate": {<key>: float, ...}      # if include_rate=True and available
        "meta": RecordMeta(...)
      }

    Notes from README:
      - Group ID mapping is defined in Table 3. (See citations in your README.)
      - Niccomo IMP is raw; derivative dZ/dt can be computed via np.gradient.
    """

    # Group ID mapping (from README Table 3).
    GROUPS: Mapping[str, str] = {
        # Niccomo
        "ecg_niccomo": "_030",
        "imp": "_031",
        "rpeaks_niccomo": "_032",
        "avo_niccomo": "_033",
        "pep_niccomo": "_034",   # called "PPEjec" in the table
        "avc_niccomo": "_035",
        "lvet_niccomo": "_036",
        # Stethoscope
        "ecg_steth": "_060",
        "pcg": "_061",
        "rpeaks_steth": "_062",
        "avo_steth": "_063",
        "pep_steth": "_064",
        "avc_steth": "_065",
        "lvet_steth": "_066",
        # Echocardiogram device
        "ecg_echo": "_090",
        "echo": "_091",
        "rpeaks_echo": "_092",
        "avo_echo": "_093",
        "pep_echo": "_094",
        "avc_echo": "_095",
        "lvet_echo": "_096",
        # PPG device
        "ecg_ppg": "_120",
        "ppg": "_121",
        "rpeaks_ppg": "_122",
        "avo_ppg": "_123",
        "pep_ppg": "_124",
        "avc_ppg": "_125",
        "lvet_ppg": "_126",
    }

    _FNAME_RE = re.compile(r"^(?P<subj>[^_]+)_(?P<exp>\d+)_(?P<rec>s\d+)\.h5$")

    def __init__(
        self,
        root: Union[str, Path],
        files: Optional[Sequence[Union[str, Path]]] = None,
        *,
        inputs: Sequence[str] = ("echo",),
        include_time: bool = True,
        include_rate: bool = False,
        compute_dzdt_from_imp: bool = False,
        echo_transpose_to_image: bool = True,
        strict: bool = False,
        dtype: torch.dtype = torch.float32,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Args:
            root: Dataset root directory. If `files` is None, scans for **/measure/*.h5.
            files: Optional explicit list of .h5 paths (absolute or relative to root).
            inputs: Keys from GROUPS to load as inputs (x).
            include_time: If True, also load per-signal time arrays when present.
            include_rate: If True, try to load sampling rate from each group ("Rate").
            compute_dzdt_from_imp: If True, adds x["dzdt"] computed from raw IMP and its time.
            echo_transpose_to_image: If True, transpose ECHO from (3,time,depth) -> (3,depth,time).
            strict: If True, missing groups raise KeyError; else missing groups become None.
            dtype: Torch dtype for numeric tensors.
            transform: Optional callable applied to the returned sample dict.
        """
        self.root = Path(root)
        self.inputs = tuple(inputs)
        self.include_time = include_time
        self.include_rate = include_rate
        self.compute_dzdt_from_imp = compute_dzdt_from_imp
        self.echo_transpose_to_image = echo_transpose_to_image
        self.strict = strict
        self.dtype = dtype
        self.transform = transform

        if files is None:
            # Typical layout: <exp_id>/measure/*.h5
            self.files: List[Path] = sorted(self.root.glob("**/measure/*.h5"))
        else:
            self.files = []
            for p in files:
                p = Path(p)
                self.files.append(p if p.is_absolute() else (self.root / p))
            self.files = sorted(self.files)

        if not self.files:
            raise FileNotFoundError(
                f"No .h5 files found. root={self.root!s}. "
                "Expected something like <root>/<experiment_id>/measure/*.h5"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        meta = self._parse_meta(path)

        x: Dict[str, Optional[torch.Tensor]] = {}
        t: Dict[str, Optional[torch.Tensor]] = {}
        rate: Dict[str, Optional[float]] = {}

        with h5py.File(path, "r") as f:
            for key in self.inputs:
                data, time, sr = self._load_key(f, key)
                x[key] = data
                if self.include_time:
                    t[key] = time
                if self.include_rate and sr is not None:
                    rate[key] = sr

        # Optional: compute dZ/dt from raw IMP (Niccomo).
        if self.compute_dzdt_from_imp:
            imp = x.get("imp")
            imp_t = t.get("imp") if self.include_time else None
            if imp is None or imp_t is None:
                if self.strict:
                    raise KeyError("compute_dzdt_from_imp=True requires 'imp' and its time vector.")
                x["dzdt"] = None
            else:
                # torch.gradient exists, but to keep behavior close to README’s np.gradient, do it in numpy:
                imp_np = imp.detach().cpu().numpy()
                t_np = imp_t.detach().cpu().numpy()
                dt_np = float(np.mean(np.diff(t_np)))
                dzdt_np = np.gradient(imp_np, dt_np)
                x["dzdt"] = torch.as_tensor(dzdt_np, dtype=self.dtype)

        sample = {"x": x, "meta": meta}
        if self.include_time:
            sample["t"] = t
        if self.include_rate:
            sample["rate"] = rate

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _parse_meta(self, path: Path) -> RecordMeta:
        m = self._FNAME_RE.match(path.name)
        if not m:
            return RecordMeta(None, None, None, str(path))
        return RecordMeta(
            subject_id=m.group("subj"),
            experiment_id=m.group("exp"),
            record_id=m.group("rec"),
            path=str(path),
        )

    def _load_key(
        self, f: h5py.File, key: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
        """
        Returns: (data_tensor_or_None, time_tensor_or_None, sampling_rate_or_None)
        """
        if key not in self.GROUPS:
            raise KeyError(f"Unknown key '{key}'. Valid keys: {sorted(self.GROUPS.keys())}")

        gid = self.GROUPS[key]
        try:
            grp = f["measure"]["value"][gid]["value"]
        except Exception as e:
            if self.strict:
                raise KeyError(f"Missing group {gid} for key '{key}' in file.") from e
            return None, None, None

        data = None
        time = None
        sr = None

        # Data
        try:
            data_arr = np.array(grp["data"]["value"])
            data_arr = np.squeeze(data_arr)  # most signals are (1, time)
            # ECHO preprocessing: stored as (3, time, depth/velocity)
            if key == "echo" and self.echo_transpose_to_image and data_arr.ndim == 3:
                # (3, time, depth) -> (3, depth, time) (image-like)
                data_arr = np.transpose(data_arr, (0, 2, 1))
            data = torch.as_tensor(data_arr, dtype=self.dtype)
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"Failed reading data for {gid} ({key}).") from e
            data = None

        # Time (not always present for scalar parameters like PEP/LVET)
        try:
            if "time" in grp:
                time_arr = np.array(grp["time"]["value"])
                time_arr = np.squeeze(time_arr)
                time = torch.as_tensor(time_arr, dtype=self.dtype)
        except Exception:
            time = None

        # Sampling rate, if present as a 'Rate' group inside the signal group.
        try:
            for k in grp.keys():
                if str(k).lower() == "rate":
                    sr_arr = np.array(grp[k]["value"])
                    sr_arr = np.squeeze(sr_arr)
                    sr = float(sr_arr)
                    break
        except Exception:
            sr = None

        return data, time, sr


if __name__ == "__main__":
    # ---- Example usage ----
    root = "./data/heartcycle"  # change to your dataset root

    ds = HeartcycleDataset(
        root=root,
        inputs=("imp", "ecg_niccomo", "echo"),
        include_time=True,
        include_rate=True,
        compute_dzdt_from_imp=True,
        echo_transpose_to_image=True,
    )

    print("=" * 60)
    print("HeartcycleDataset sanity check")
    print("=" * 60)
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]

    print("\n--- META ---")
    print(sample["meta"])

    print("\n--- INPUTS (x) ---")
    for k, v in sample["x"].items():
        if v is None:
            print(f"{k:15s}: None")
        else:
            print(f"{k:15s}: shape={tuple(v.shape)}, dtype={v.dtype}")
            print(f"  first values: {v.flatten()[:5]}")

    if "t" in sample:
        print("\n--- TIME VECTORS (t) ---")
        for k, v in sample["t"].items():
            if v is None:
                print(f"{k:15s}: None")
            else:
                print(f"{k:15s}: shape={tuple(v.shape)}")
                print(f"  first times: {v[:5]}")

    if "rate" in sample:
        print("\n--- SAMPLING RATES ---")
        for k, v in sample["rate"].items():
            print(f"{k:15s}: {v}")

    print("\nDone ✔")
