from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def pretrain_collate(
    batch: List[Dict[str, Any]],
    *,
    max_frames: int,
    augmentations: Optional[torch.nn.Module] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      {
        "video": Tensor [B, max_frames, C, H, W],
        "timestamps": Tensor [B, max_frames],  # shifted so each sample starts at 0
      }

    Per-sample:
      - If T > max_frames: random subset of frames (without replacement), keep time order.
      - If T < max_frames: insert k frames by sampling k timestamps uniformly in (tmin,tmax)
        and linearly interpolating between temporal neighbors.
      - Always sorts by timestamp and shifts timestamps by -min so they start at 0.
    """
    if max_frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {max_frames}")

    vids, tss, augvs = [], [], []

    for s in batch:
        v = s["video"]         # [T,C,H,W]
        ts = s["timestamps"]   # [T]

        if v.dim() != 4:
            raise ValueError(f"video must be [T,C,H,W], got {tuple(v.shape)}")
        if ts.dim() != 1:
            raise ValueError(f"timestamps must be [T], got {tuple(ts.shape)}")
        if v.size(0) != ts.numel():
            raise ValueError(f"T mismatch: video T={v.size(0)} vs timestamps={ts.numel()}")

        T, C, H, W = v.shape

        # --- explicit handling for empty / single-frame clips ---
        if T == 0:
            raise ValueError("video has T==0 frames; cannot collate/resize an empty clip.")
        if T == 1:
            # Repeat the single frame to max_frames; timestamps become all zeros.
            v = v[:1].expand(max_frames, -1, -1, -1).contiguous()
            ts = torch.zeros((max_frames,), device=v.device, dtype=torch.float32)
            vids.append(v)
            tss.append(ts)
            continue
        # --------------------------------------------------------

        ts = ts.to(device=v.device, dtype=torch.float32)

        # enforce time order (just in case)
        order = torch.argsort(ts)
        v = v.index_select(0, order)
        ts = ts.index_select(0, order)

        # update after sorting (T unchanged, but keep logic clear)
        t0, t1 = ts[0], ts[-1]

        if T > max_frames:
            idx = torch.randperm(T, device=v.device, generator=generator)[:max_frames]
            idx, _ = torch.sort(idx)  # keep temporal order
            v = v.index_select(0, idx)
            ts = ts.index_select(0, idx)

        elif T < max_frames:
            k = max_frames - T

            # --- explicit endpoint/degenerate-span handling ---
            if t1 <= t0:
                # No usable time span to sample within; just repeat frames to reach max_frames.
                reps = (max_frames + T - 1) // T  # ceil
                v = v.repeat(reps, 1, 1, 1)[:max_frames]
                ts = ts.repeat(reps)[:max_frames]
            else:
                # sample k timestamps in (t0,t1) (practically [t0,t1) due to rand)
                new_ts = t0 + torch.rand(k, device=v.device, generator=generator) * (t1 - t0)

                ts_all = torch.cat([ts, new_ts], dim=0)
                order = torch.argsort(ts_all)
                ts_sorted = ts_all.index_select(0, order)

                # interpolate frames for each target time in ts_sorted
                right = torch.searchsorted(ts, ts_sorted, right=False).clamp(0, T - 1)
                left = (right - 1).clamp(0, T - 1)

                tL = ts[left]
                tR = ts[right]
                denom = (tR - tL)

                w = torch.zeros_like(ts_sorted)
                m = denom > 0
                w[m] = (ts_sorted[m] - tL[m]) / denom[m]
                w = w.view(-1, 1, 1, 1)

                v = v[left] * (1.0 - w) + v[right] * w
                ts = ts_sorted
            # ---------------------------------------------------

        # Augmentations
        if augmentations is not None:
            aug_v = augmentations(v)
            aug_v = aug_v * 2 - 1
            augvs.append(aug_v)

            # timestamp jitter
            fps = s["metadata"]["FPS"]
            jitter = (torch.rand_like(ts) - 0.5) * (0.5 / fps)  # up to ±0.25 frame jitter
            ts = ts + jitter

        # shift timestamps to start at 0
        ts = ts - ts[0]

        # Normalize [0, 1] to [-1, 1]
        v = v * 2 - 1

        vids.append(v)
        tss.append(ts)

    out = {
        "video": torch.stack(vids, dim=0),       # [B,max_frames,C,H,W]
        "timestamps": torch.stack(tss, dim=0),   # [B,max_frames]
    }
    if augmentations is not None:
        out["aug_video"] = torch.stack(augvs, dim=0)  # [B,max_frames,C,H,W]
    return out


def EF_collate(batch: List[Dict[str, Any]],
    *,
    max_frames: int,
    augmentations: Optional[torch.nn.Module] = None,
    generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
    out = pretrain_collate(batch, max_frames=max_frames, augmentations=augmentations, generator=generator)

    ef = [s["metadata"]["EF"] for s in batch]
    ef = torch.tensor(ef, dtype=torch.float32, device=out["video"].device)
    ef = ef / 100.0  # Normalize EF to [0, 1]
    out["EF"] = ef
    return out


def EDES_collate(batch):
    videos = torch.stack([sample["video"] for sample in batch], dim=0)          # [B,T,C,H,W]
    videos = videos * 2 - 1  # [0,1] → [-1,1]
    timestamps = torch.stack([sample["timestamps"] for sample in batch], dim=0) # [B,T]

    frames_idx = []; fps_list = []
    for sample in batch:
        fi = sample["masks"]["frame_indices"]
        if fi is None:
            fi = torch.empty((0,), dtype=torch.long)
        else:
            fi = fi.long()
        frames_idx.append(fi)
        fps_list.append(sample["metadata"]["FPS"])
        
    return videos, timestamps, frames_idx, fps_list



def LV_collate(batch, max_frames: int, augmentations=None, generator=None):
    if max_frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {max_frames}")

    vids, tss, augvs, masks, key_idx = [], [], [], [], []

    for s in batch:
        v = s["video"]  # [T,C,H,W]
        ts0 = s["timestamps"]  # [T]
        ma = s["masks"]["masks"].unsqueeze(1)  # [2,1,H,W]
        fi = s["masks"]["frame_indices"].to(device=v.device, dtype=torch.long)  # [2]

        T, C, H, W = v.shape
        ts = ts0.to(device=v.device, dtype=torch.float32)

        # ---- enforce time order AND remap fi accordingly ----
        order = torch.argsort(ts)
        v = v.index_select(0, order)
        ts = ts.index_select(0, order)

        # remap fi from old indexing -> new indexing after sort
        inv = torch.empty_like(order)
        inv[order] = torch.arange(T, device=v.device, dtype=torch.long)
        fi = inv[fi]  # still [2]

        # keep for later (used only when we need original timestamps)
        t0, t1 = ts[0], ts[-1]

        # ---- adjust length to max_frames while keeping key frames ----
        if T > max_frames:
            # sample subset, then force-include key frames (in sorted-index space)
            idx = torch.randperm(T, device=v.device, generator=generator)[:max_frames]
            idx = idx.to(dtype=torch.long)

            fi_in_idx = torch.isin(fi, idx)
            if not fi_in_idx.all():
                missing = fi[~fi_in_idx]  # 1 or 2 elements
                replace_pos = torch.randperm(max_frames, device=v.device, generator=generator)[:missing.numel()]
                idx[replace_pos] = missing

            idx, _ = torch.sort(idx)  # temporal order
            v = v.index_select(0, idx)
            ts = ts.index_select(0, idx)

            # map fi -> positions inside idx (idx sorted, fi guaranteed included)
            # searchsorted gives insertion point; since fi included, idx[pos]==fi
            pos = torch.searchsorted(idx, fi)
            fi = pos.to(dtype=torch.long)  # [2]

        elif T < max_frames:
            k = max_frames - T

            # sample k timestamps in [t0, t1)
            new_ts = t0 + torch.rand(k, device=v.device, generator=generator) * (t1 - t0)

            ts_all = torch.cat([ts, new_ts], dim=0)  # [T+k]
            order_all = torch.argsort(ts_all)         # permutation of [0..T+k-1]
            ts_sorted = ts_all.index_select(0, order_all)  # [max_frames]

            # interpolate frames at ts_sorted
            right = torch.searchsorted(ts, ts_sorted, right=False).clamp(0, T - 1)
            left = (right - 1).clamp(0, T - 1)

            tL = ts[left]
            tR = ts[right]
            denom = (tR - tL)

            w = torch.zeros_like(ts_sorted)
            m = denom > 0
            w[m] = (ts_sorted[m] - tL[m]) / denom[m]
            w = w.view(-1, 1, 1, 1)

            v = v[left] * (1.0 - w) + v[right] * w
            ts = ts_sorted

            # robust fi remap without float equality:
            # positions of original frames (0..T-1 from ts_all) within the sorted array
            pos_of_original = torch.nonzero(order_all < T, as_tuple=False).squeeze(1)  # [T]
            fi = pos_of_original[fi].to(dtype=torch.long)  # [2]

        # ---- augmentations ----
        if augmentations is not None:
            aug_v = augmentations(v)
            aug_v = aug_v * 2 - 1
            augvs.append(aug_v)

            fps = s["metadata"]["FPS"]
            jitter = (torch.rand_like(ts) - 0.5) * (0.5 / fps)  # ±0.25 frame jitter
            ts = ts + jitter

        # ---- normalize / shift time ----
        ts = ts - ts[0]
        v = v * 2 - 1

        vids.append(v)
        tss.append(ts)
        masks.append(ma)
        key_idx.append(fi)

    out = {
        "video": torch.stack(vids, dim=0),             # [B,max_frames,C,H,W]
        "timestamps": torch.stack(tss, dim=0),         # [B,max_frames]
        "masks": torch.stack(masks, dim=0),            # [B,2,1,H,W]
        "frame_indices": torch.stack(key_idx, dim=0),  # [B,2]
    }
    if augmentations is not None:
        out["aug_video"] = torch.stack(augvs, dim=0)   # [B,max_frames,C,H,W]
    return out



def AE_collate(batch, max_frames, augmentations=None, time_jitter=False, generator=None):
    vs, tss = [], []

    for s in batch:
        v = s["video"]        # [T, C, H, W] in [0, 1]
        ts = s["timestamps"]  # [T] in seconds
        T = v.shape[0]
        device = v.device

        # Sampling
        if T <= max_frames:
            idx = torch.randint(0, T, (max_frames,), device=device, generator=generator)
        else:
            idx = torch.randperm(T, device=device, generator=generator)[:max_frames]

        idx, _ = torch.sort(idx)  # keep temporal order

        # Gather frames/timestamps
        v = v.index_select(0, idx)  # [max_frames, C, H, W]
        ts = ts.index_select(0, idx) # [max_frames]

        # Shift timestamps to start at 0
        ts = ts - ts[0]

        if time_jitter: # ±0.5fps timestamp jitter for middle timestamps
            fps = s["metadata"]["FPS"]
            ts[1:-1] = ts[1:-1] + (torch.rand_like(ts[1:-1]) - 0.5) * (1 / fps)

        # Augmentations (only on in_frames)
        if augmentations is not None:
            v = augmentations(v)

        vs.append(v)
        tss.append(ts)

    # Stack
    videos = torch.stack(vs, dim=0)       # [B, max_frames, C, H, W]
    timestamps = torch.stack(tss, dim=0)  # [B, max_frames

    # Normalize to [-1, 1]
    return {
        "videos": videos.mul_(2).sub_(1),
        "timestamps": timestamps,
    }