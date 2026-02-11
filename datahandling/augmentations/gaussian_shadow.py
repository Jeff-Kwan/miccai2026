"""
Gaussian shadow (attenuation) augmentation for a *single video* tensor [T, C, H, W].

- Samples ONE Gaussian attenuation field per call and applies it to all frames,
  so frames from the same video share the same attenuation.
- Expects input float in [0, 1], returns float in [0, 1].
- Torchvision-style transform module (callable) that composes with other transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


FloatOrRange = Union[float, Tuple[float, float], Sequence[float]]


def _sample_range(v: FloatOrRange, *, device: torch.device) -> float:
    if isinstance(v, (tuple, list)) and len(v) == 2:
        lo, hi = float(v[0]), float(v[1])
        r = torch.rand((), device=device).item()
        return lo + (hi - lo) * r
    return float(v)


class GaussianShadowVideo:
    """
    Torchvision-compatible callable transform.

    Parameters
    ----------
    strength:
        Attenuation strength. If range (lo, hi), sampled uniformly.
        The attenuation field is: shadow = 1 - strength * exp(-gaussian).
        Typical range: (0.25, 0.8)

    sigma_x, sigma_y:
        Gaussian std as *fraction* of width/height. If range, sampled uniformly.
        Typical range: (0.01, 0.2)

    p:
        Probability of applying the transform.

    clamp:
        Whether to clamp output back to [0, 1] (recommended).
    """
    def __init__(
        self,
        strength: FloatOrRange = (0.25, 0.8),
        sigma_x: FloatOrRange = (0.01, 0.2),
        sigma_y: FloatOrRange = (0.01, 0.2),
        p: float = 0.5,
        clamp: bool = True):
        self.strength = strength
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.p = p
        self.clamp = clamp

    def __call__(self, video: Tensor, scan_mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        video:
            Tensor [T, C, H, W], float in [0, 1].
        scan_mask:
            Optional Tensor [H, W] (or broadcastable to [H, W]) where 1 means
            "apply attenuation" and 0 means "leave unchanged". If None, uses all ones.

        Returns
        -------
        Tensor [T, C, H, W], float in [0, 1].
        """
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"video must be a torch.Tensor, got {type(video)}")
        if video.ndim != 4:
            raise ValueError(f"Expected video shape [T, C, H, W], got {tuple(video.shape)}")
        if video.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"Expected float tensor in [0,1], got dtype {video.dtype}")

        # Apply with probability p
        if self.p < 1.0 and torch.rand((), device=video.device).item() > self.p:
            return video

        T, C, H, W = video.shape
        device = video.device

        if scan_mask is None:
            mask_hw = torch.ones((H, W), device=device, dtype=video.dtype)
        else:
            if not isinstance(scan_mask, torch.Tensor):
                raise TypeError(f"scan_mask must be a torch.Tensor, got {type(scan_mask)}")
            # make [H, W]
            if scan_mask.ndim == 2:
                mask_hw = scan_mask
            else:
                # allow broadcastable shapes like [1,H,W] or [H,W,1] etc.
                mask_hw = scan_mask.squeeze()
                if mask_hw.ndim != 2:
                    raise ValueError(f"scan_mask must be [H,W] (or squeeze to it), got {tuple(scan_mask.shape)}")
            if mask_hw.shape != (H, W):
                raise ValueError(f"scan_mask shape must match [H,W]=({H},{W}), got {tuple(mask_hw.shape)}")
            mask_hw = mask_hw.to(device=device, dtype=video.dtype)

        # Sample parameters ONCE per video (shared across frames)
        strength = _sample_range(self.strength, device=device)
        sx = _sample_range(self.sigma_x, device=device) * W
        sy = _sample_range(self.sigma_y, device=device) * H

        # Avoid degenerate sigmas
        sx = max(sx, 1e-6)
        sy = max(sy, 1e-6)

        # Random center (pixel coords)
        mu_x = torch.randint(low=0, high=W, size=(), device=device).to(torch.float32)
        mu_y = torch.randint(low=0, high=H, size=(), device=device).to(torch.float32)

        # Build Gaussian attenuation field [H, W]
        x = torch.arange(W, device=device, dtype=torch.float32)
        y = torch.arange(H, device=device, dtype=torch.float32)
        yv, xv = torch.meshgrid(y, x, indexing="ij")

        gauss = torch.exp(
            -(((xv - mu_x) ** 2) / (2.0 * (sx ** 2)) + ((yv - mu_y) ** 2) / (2.0 * (sy ** 2)))
        )
        shadow = 1.0 - float(strength) * gauss  # [H, W], float32
        shadow = shadow.to(dtype=video.dtype)

        # Apply only inside mask; outside mask, keep multiplier = 1
        shadow = torch.where(mask_hw > 0, shadow, torch.ones_like(shadow))

        # Apply to all frames/channels (broadcast [1,1,H,W])
        out = video * shadow.view(1, 1, H, W)

        if self.clamp:
            out = out.clamp_(0.0, 1.0)

        return out