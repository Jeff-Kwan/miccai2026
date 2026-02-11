"""
Video-domain intensity and corruption augmentations for training-time regularization.

All transforms operate on a single video tensor of shape [T, C, H, W] with
floating-point values in the range [0, 1], and return the same format.

The chain introduces realistic acquisition and preprocessing variability by
applying, per video: gamma adjustment, brightness/contrast jitter, multiplicative
speckle noise, random frame dropout, and spatiotemporal erasing. Random parameters
are sampled once per video so that temporal consistency is preserved across frames.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor

FloatRange = Union[float, Tuple[float, float]]


def _sample_uniform_torch(rng: FloatRange, device: torch.device) -> float:
    """Sample uniformly using torch (plays nicer with torch seeding)."""
    if isinstance(rng, (tuple, list)):
        lo, hi = float(rng[0]), float(rng[1])
        return float((lo + (hi - lo) * torch.rand((), device=device)).item())
    return float(rng)


class RandomGammaVideo(nn.Module):
    """
    Random gamma correction for a single video: [T, C, H, W], float in [0, 1].
    Samples ONE gamma per video so all frames share the same adjustment.
    """

    def __init__(self, gamma: FloatRange = (0.7, 1.5), p: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W], got {tuple(x.shape)}")
        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"Expected float tensor, got dtype={x.dtype}")

        if self.p < 1.0 and torch.rand((), device=x.device).item() > self.p:
            return x

        g = _sample_uniform_torch(self.gamma, x.device)
        return x.clamp(0.0, 1.0).pow(g).clamp(0.0, 1.0)


class ClipBrightnessContrastVideo(nn.Module):
    """
    Brightness/contrast jitter for a single video: [T, C, H, W], float in [0, 1].
    Samples ONE brightness and ONE contrast per video so frames match.
    """

    def __init__(self, brightness: float = 0.3, contrast: float = 0.2, p: float = 1.0) -> None:
        super().__init__()
        self.b = float(brightness)
        self.c = float(contrast)
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W], got {tuple(x.shape)}")
        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"Expected float tensor, got dtype={x.dtype}")

        if self.p < 1.0 and torch.rand((), device=x.device).item() > self.p:
            return x

        b = float(torch.empty((), device=x.device).uniform_(-self.b, self.b).item())
        c = float(torch.empty((), device=x.device).uniform_(1.0 - self.c, 1.0 + self.c).item())

        # Mean over spatial dims, per (T,C)
        mean = x.mean(dim=(-2, -1), keepdim=True)  # [T,C,1,1]
        y = (x - mean) * c + mean + b
        return y.clamp(0.0, 1.0)


class SpeckleNoiseVideo(nn.Module):
    """
    Multiplicative speckle noise for a single video: [T, C, H, W], float in [0, 1].
    Samples ONE std per video so all frames share the same noise strength.
    """

    def __init__(self, std: FloatRange = (0.02, 0.1), p: float = 1.0) -> None:
        super().__init__()
        self.std = std
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W], got {tuple(x.shape)}")
        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"Expected float tensor, got dtype={x.dtype}")

        if self.p < 1.0 and torch.rand((), device=x.device).item() > self.p:
            return x

        std = _sample_uniform_torch(self.std, x.device)
        noise = torch.randn_like(x) * std
        y = x + x * noise
        return y.clamp(0.0, 1.0)


class FrameDropoutVideo(nn.Module):
    """
    Randomly drops entire frames with probability p.
    Input:  [T, C, H, W] float in [0, 1]
    Output: [T, C, H, W] float in [0, 1]
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W], got {tuple(x.shape)}")
        T = x.shape[0]
        keep = (torch.rand(T, device=x.device) > self.p).to(x.dtype)  # [T]
        keep = keep[:, None, None, None]  # [T,1,1,1]
        return (x * keep).clamp(0.0, 1.0)


class RandomVideoErasing(nn.Module):
    """
    Randomly fills one or more spatiotemporal cuboids in a single video tensor.

    Input:  x of shape [T, C, H, W], float in [0, 1]
    Output: same shape, with K cuboids set to `value` (across all C).
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.4),   # fraction of H*W
        ratio: Tuple[float, float] = (0.3, 3.3),    # w/h
        t_scale: Tuple[float, float] = (0.1, 1.0),  # fraction of T
        value: float = 0.0,
        attempts: int = 10,
        inplace: bool = False,
        num_cuboids: int = 1,
        num_cuboids_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        if scale[0] <= 0 or scale[1] <= 0 or scale[0] > scale[1]:
            raise ValueError("scale must be (min, max) with 0 < min <= max.")
        if ratio[0] <= 0 or ratio[0] > ratio[1]:
            raise ValueError("ratio must be (min, max) with 0 < min <= max.")
        if t_scale[0] <= 0 or t_scale[1] <= 0 or t_scale[0] > t_scale[1]:
            raise ValueError("t_scale must be (min, max) with 0 < min <= max.")
        if attempts < 1:
            raise ValueError("attempts must be >= 1.")
        if num_cuboids_range is not None:
            lo, hi = num_cuboids_range
            if lo < 0 or hi < lo:
                raise ValueError("num_cuboids_range must be (low, high) with 0 <= low <= high.")
        else:
            if num_cuboids < 0:
                raise ValueError("num_cuboids must be >= 0.")

        self.p = float(p)
        self.scale = scale
        self.ratio = ratio
        self.t_scale = t_scale
        self.value = float(value)
        self.attempts = int(attempts)
        self.inplace = bool(inplace)
        self.num_cuboids = int(num_cuboids)
        self.num_cuboids_range = num_cuboids_range

    @staticmethod
    def _rand_uniform(a: float, b: float) -> float:
        return a + (b - a) * random.random()

    def _sample_k(self) -> int:
        if self.num_cuboids_range is None:
            return self.num_cuboids
        lo, hi = self.num_cuboids_range
        return random.randint(lo, hi)

    def _sample_region(self, T: int, H: int, W: int):
        """
        Returns (t0, t1, y0, y1, x0, x1) or None if couldn't find a valid region.
        """
        area = H * W

        for _ in range(self.attempts):
            t_frac = self._rand_uniform(self.t_scale[0], self.t_scale[1])
            t_len = max(1, int(round(t_frac * T)))
            if t_len > T:
                continue

            erase_area = self._rand_uniform(self.scale[0], self.scale[1]) * area
            aspect = self._rand_uniform(self.ratio[0], self.ratio[1])

            h = int(round(math.sqrt(erase_area / aspect)))
            w = int(round(math.sqrt(erase_area * aspect)))
            if h < 1 or w < 1 or h > H or w > W:
                continue

            t0 = random.randint(0, T - t_len)
            y0 = random.randint(0, H - h)
            x0 = random.randint(0, W - w)
            return (t0, t0 + t_len, y0, y0 + h, x0, x0 + w)

        return None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W], got {tuple(x.shape)}")
        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"Expected float tensor, got dtype={x.dtype}")

        if (not self.training) or (random.random() > self.p):
            return x

        T, C, H, W = x.shape
        out = x if self.inplace else x.clone()

        K = self._sample_k()
        for _ in range(K):
            region = self._sample_region(T, H, W)
            if region is None:
                continue
            t0, t1, y0, y1, x0, x1 = region
            out[t0:t1, :, y0:y1, x0:x1] = self.value

        return out.clamp(0.0, 1.0)