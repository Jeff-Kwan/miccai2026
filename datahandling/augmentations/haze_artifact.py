from __future__ import annotations

from typing import Tuple, Union

import torch


NumberOrRange = Union[float, Tuple[float, float]]


class VideoHazeArtifact(torch.nn.Module):
    """
    Simple training-time haze artifact for a single video clip.

    - Input:  video tensor of shape [T, C, H, W], float in [0, 1]
    - Output: same shape, float in [0, 1]
    - Same attenuation/haze field is applied to all frames in the clip.

    Designed to compose with torchvision transforms (i.e. callable module).
    """
    def __init__(
        self,
        radius: NumberOrRange = (0.05, 0.95),
        sigma: NumberOrRange = (0.01, 0.10),  # avoid exactly 0 to prevent div-by-zero
        strength: float = 0.5,                # scales haze contribution
        p: float = 0.5,                       # probability of applying
    ):
        super().__init__()
        self.radius = radius
        self.sigma = sigma
        self.strength = strength
        self.p = p

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(video):
            raise TypeError(f"Expected torch.Tensor, got {type(video)}")
        if video.ndim != 4:
            raise ValueError(f"Expected shape [T, C, H, W], got {tuple(video.shape)}")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be in [0, 1]")

        # No-op with probability (1 - p)
        if self.p < 1.0 and torch.rand((), device=video.device) >= self.p:
            return video

        # Work in float32 for stable math; preserve device
        v = video.to(dtype=torch.float32)
        T, C, H, W = v.shape

        haze = self._generate_haze_field(
            H=H, W=W, device=v.device, dtype=v.dtype
        )  # [H, W], shared across frames

        # Broadcast to [T, C, H, W]
        haze = haze.view(1, 1, H, W)

        out = v + self.strength * haze
        out = out.clamp_(0.0, 1.0)

        # Return float in [0, 1]. (Keep float32; most video pipelines use float.)
        return out

    def _sample_range(self, x: NumberOrRange, device: torch.device) -> float:
        if isinstance(x, (tuple, list)):
            lo, hi = float(x[0]), float(x[1])
            r = torch.empty((), device=device).uniform_(lo, hi).item()
            return r
        return float(x)

    @torch.no_grad()
    def _generate_haze_field(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Coordinate grid in [0, 1]
        x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        yv, xv = torch.meshgrid(y, x, indexing="ij")  # [H, W]

        # Same geometry as your numpy version: center at x=0.5, y=0.0
        r = torch.sqrt((xv - 0.5) ** 2 + (yv - 0.0) ** 2)

        haze_radius = self._sample_range(self.radius, device=device)
        haze_sigma = self._sample_range(self.sigma, device=device)
        haze_sigma = max(haze_sigma, 1e-6)

        # Random haze texture, gated by a radial Gaussian ring
        noise = torch.rand((H, W), device=device, dtype=dtype)
        gate = torch.exp(-((r - haze_radius) ** 2) / (2.0 * (haze_sigma**2)))

        haze = noise * gate
        return haze