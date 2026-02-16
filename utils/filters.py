import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.signal import butter
from torchaudio.functional import filtfilt

def highpass(signal: torch.Tensor, fs: float, cutoff: float = 0.5, order: int = 4):
    """
    signal: [T, D]
    fs: sampling rate (Hz)
    cutoff: cutoff frequency (Hz)
    order: Butterworth filter order
    """
    signal = signal.transpose(0, 1)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # Design filter (NumPy/SciPy step just computes coefficients)
    b, a = butter(order, normal_cutoff, btype="high", analog=False)

    # Move coefficients to torch, matching dtype/device of the signal
    b = torch.tensor(b, dtype=signal.dtype, device=signal.device)
    a = torch.tensor(a, dtype=signal.dtype, device=signal.device)

    # torchaudio expects shape (..., time)
    filtered = filtfilt(signal, a_coeffs=a, b_coeffs=b)
    return filtered.transpose(0, 1)



class SavGolFilterTime(nn.Module):
    """
    Savitzky–Golay filter along the *time* dimension T for tensors shaped:
      - [T, D]
      - [B, T, D]

    Implemented as a fixed 1D grouped convolution, so it runs on GPU.

    Args:
      window_length: odd int >= 3
      polyorder: int, 0 <= polyorder < window_length
      deriv: derivative order (0 = smoothing)
      delta: sample spacing for derivative scaling
      pad_mode: 'reflect' | 'replicate' | 'circular' | 'constant'
      pad_value: only used for pad_mode == 'constant'
      dtype: kernel dtype used when computing coefficients (float32 typical)
    """

    def __init__(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 0,
        delta: float = 1.0,
        pad_mode: str = "reflect",
        pad_value: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if window_length % 2 != 1 or window_length < 3:
            raise ValueError("window_length must be an odd integer >= 3.")
        if polyorder < 0 or polyorder >= window_length:
            raise ValueError("polyorder must be >= 0 and < window_length.")
        if deriv < 0:
            raise ValueError("deriv must be >= 0.")
        if deriv > polyorder:
            raise ValueError("deriv must be <= polyorder.")
        if delta <= 0:
            raise ValueError("delta must be > 0.")
        if pad_mode not in {"reflect", "replicate", "circular", "constant"}:
            raise ValueError("pad_mode must be one of: reflect, replicate, circular, constant.")

        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.deriv = int(deriv)
        self.delta = float(delta)
        self.pad_mode = pad_mode
        self.pad_value = float(pad_value)

        coeffs = self._compute_savgol_coeffs_torch(
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            dtype=dtype,
        )
        # Stored as (K,) and reshaped in forward; moves with .to(device)
        self.register_buffer("coeffs", coeffs, persistent=True)

    @staticmethod
    def _compute_savgol_coeffs_torch(
        window_length: int,
        polyorder: int,
        deriv: int,
        delta: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        half = window_length // 2
        x = torch.arange(-half, half + 1, dtype=dtype)  # (K,)

        # Design matrix A: (K, polyorder+1), A[j, i] = x_j^i
        A = torch.stack([x ** i for i in range(polyorder + 1)], dim=1)

        # Pseudoinverse
        pinvA = torch.linalg.pinv(A)  # (polyorder+1, K)

        # Target in polynomial-coefficient space for m-th derivative at 0: m! * a_m
        b = torch.zeros(polyorder + 1, dtype=dtype)
        b[deriv] = math.factorial(deriv)

        coeffs = pinvA.T @ b  # (K,)
        coeffs = coeffs / (delta ** deriv)
        return coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Filters along T.

        Input:
          x: [T, D] or [B, T, D]
        Output:
          same shape as input
        """
        if x.dim() == 2:
            # [T, D] -> [1, D, T]
            T, D = x.shape
            x_ = x.transpose(0, 1).unsqueeze(0)  # [1, D, T]
            squeeze_batch = True
        elif x.dim() == 3:
            # [B, T, D] -> [B, D, T]
            B, T, D = x.shape
            x_ = x.transpose(1, 2)  # [B, D, T]
            squeeze_batch = False
        else:
            raise ValueError("Input must have shape [T, D] or [B, T, D].")

        K = self.window_length
        pad = K // 2

        # Pad along time dimension (last dim in [*, D, T])
        if self.pad_mode == "constant":
            x_ = F.pad(x_, (pad, pad), mode="constant", value=self.pad_value)
        else:
            x_ = F.pad(x_, (pad, pad), mode=self.pad_mode)

        # Grouped conv: apply same kernel independently to each feature channel D
        Dch = x_.shape[1]
        weight = self.coeffs.to(device=x_.device, dtype=x_.dtype).view(1, 1, K).repeat(Dch, 1, 1)
        y_ = F.conv1d(x_, weight, groups=Dch)

        # Back to original layout
        if squeeze_batch:
            # [1, D, T] -> [T, D]
            return y_.squeeze(0).transpose(0, 1)
        else:
            # [B, D, T] -> [B, T, D]
            return y_.transpose(1, 2)