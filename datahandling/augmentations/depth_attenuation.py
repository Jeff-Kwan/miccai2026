# depth_attenuation_video.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def _sample_uniform(
    x: Union[float, Tuple[float, float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(x, (tuple, list)):
        lo, hi = float(x[0]), float(x[1])
        r = torch.rand((), device=device, dtype=dtype)
        return r * (hi - lo) + lo
    return torch.tensor(float(x), device=device, dtype=dtype)


@dataclass
class _GridCache:
    # Cache per (H, W, device, dtype)
    key: Tuple[int, int, str, str] = (-1, -1, "", "")
    distances: Optional[torch.Tensor] = None  # [H, W]


class DepthAttenuationVideo(nn.Module):
    """
    Depth attenuation augmentation for a single video tensor shaped [T, C, H, W].

    - Input:  float tensor in [0, 1]
    - Output: float tensor in [0, 1] (clamped)

    Key property: all frames in the same video share the same attenuation map
    (i.e., same sampled attenuation_rate + same spatial map).
    """

    def __init__(
        self,
        attenuation_rate: Union[float, Tuple[float, float]] = (0.0, 3.0),
        max_attenuation: float = 0.0,
        p: float = 0.5,
        center_x: float = 0.5,
        center_y: float = 0.0,
        cache_grid: bool = True,
    ) -> None:
        super().__init__()
        self.attenuation_rate = attenuation_rate
        self.max_attenuation = float(max_attenuation)
        self.p = float(p)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.cache_grid = bool(cache_grid)
        self._grid_cache = _GridCache()

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [T, C, H, W], float in [0, 1]
        """
        if video.ndim != 4:
            raise ValueError(f"Expected video with shape [T,C,H,W], got {tuple(video.shape)}")
        if not torch.is_floating_point(video):
            raise TypeError("Expected a floating point tensor in [0, 1].")

        # Apply with probability p
        if self.p < 1.0:
            if torch.rand((), device=video.device).item() > self.p:
                return video

        T, C, H, W = video.shape
        device, dtype = video.device, video.dtype

        distances = self._get_distances(H, W, device=device, dtype=dtype)  # [H, W]
        rate = _sample_uniform(self.attenuation_rate, device=device, dtype=dtype)

        # Bounded exponential decay: (1 - m) * exp(-rate * d) + m
        m = torch.tensor(self.max_attenuation, device=device, dtype=dtype)
        attn = (1.0 - m) * torch.exp(-rate * distances) + m  # [H, W]

        # Broadcast: [T, C, H, W] * [1, 1, H, W]
        out = video * attn.view(1, 1, H, W)
        return out.clamp_(0.0, 1.0)

    def _get_distances(
        self,
        H: int,
        W: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.cache_grid:
            return self._make_distances(H, W, device=device, dtype=dtype)

        key = (H, W, str(device), str(dtype))
        if self._grid_cache.distances is None or self._grid_cache.key != key:
            self._grid_cache.key = key
            self._grid_cache.distances = self._make_distances(H, W, device=device, dtype=dtype)
        return self._grid_cache.distances

    def _make_distances(
        self,
        H: int,
        W: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Normalized coordinate grid:
        # x in [0,1] across width, y in [0,1] across height
        x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        yv, xv = torch.meshgrid(y, x, indexing="ij")

        # Distance from a probe-like origin near top-center by default
        dx = xv - self.center_x
        dy = yv - self.center_y
        return torch.sqrt(dx * dx + dy * dy)