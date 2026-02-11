from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from skimage.restoration import denoise_bilateral


FloatRange = Union[float, Tuple[float, float]]


class VideoSpeckleReduction(torch.nn.Module):
    """
    Bilateral-filter speckle reduction for a single video sample shaped [T, C, H, W].

    - Samples sigma_spatial and sigma_color PER FRAME (independent strength each frame).
    - Expects input float in [0, 1], returns float in [0, 1].
    - Uses scikit-image denoise_bilateral (CPU).
    """

    def __init__(
        self,
        sigma_spatial: FloatRange = (0.1, 2.0),
        sigma_color: FloatRange = (0.0, 1.0),
        window_size: int = 5,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.window_size = int(window_size)
        self.p = float(p)

    @staticmethod
    def _sample_uniform(rng: FloatRange, device: torch.device) -> float:
        if isinstance(rng, (tuple, list)):
            lo, hi = float(rng[0]), float(rng[1])
            # sample on torch to keep determinism under torch seeding
            return float((lo + (hi - lo) * torch.rand((), device=device)).item())
        return float(rng)

    def forward(self, video: Tensor) -> Tensor:
        """
        video: torch.Tensor of shape [T, C, H, W], float in [0, 1]
        returns: same shape, float in [0, 1]
        """
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(video)}")
        if video.ndim != 4:
            raise ValueError(f"Expected video shape [T, C, H, W], got {tuple(video.shape)}")
        if video.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"Expected float tensor, got dtype={video.dtype}")

        # Apply with probability p (per video)
        if self.p < 1.0 and torch.rand((), device=video.device).item() > self.p:
            return video

        # skimage runs on CPU and expects numpy arrays
        video_cpu = video.detach().to("cpu")
        out = torch.empty_like(video_cpu)

        # Process each frame with independently sampled sigmas
        # Convert [C, H, W] -> [H, W, C] for skimage (channel_axis=-1)
        T, C, H, W = video_cpu.shape
        for t in range(T):
            # Sample PER FRAME (independent attenuation)
            sigma_spatial = self._sample_uniform(self.sigma_spatial, device=video.device)
            sigma_color = self._sample_uniform(self.sigma_color, device=video.device)

            frame_chw = video_cpu[t]
            frame_hwc = frame_chw.permute(1, 2, 0).numpy()  # float, [0,1]

            den = denoise_bilateral(
                frame_hwc,
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                win_size=self.window_size,
                channel_axis=-1,
            )

            den_t = torch.from_numpy(den).permute(2, 0, 1)  # back to [C,H,W]
            out[t] = den_t

        # Ensure range [0,1] and move back to original device/dtype
        return out.clamp_(0.0, 1.0).to(device=video.device, dtype=video.dtype)