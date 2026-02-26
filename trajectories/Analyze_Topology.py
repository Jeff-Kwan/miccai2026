from tqdm import tqdm
import numpy as np
from ripser import ripser
from Dataset import get_latents_dataset
from scipy.signal import detrend

train_ds, val_ds, test_ds = get_latents_dataset()


def _subsample_time(z: np.ndarray, max_points: int = 256) -> np.ndarray:
    T = z.shape[0]
    if T <= max_points:
        return z
    idx = np.linspace(0, T - 1, max_points).astype(int)
    return z[idx]


def _traj_scale(z: np.ndarray, sample_pairs: int = 512, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    T = z.shape[0]
    if T < 2:
        return 1.0
    i = rng.integers(0, T, size=sample_pairs)
    j = rng.integers(0, T, size=sample_pairs)
    d = np.linalg.norm(z[i] - z[j], axis=1)
    s = float(np.median(d))
    return s if s > 1e-12 else 1.0


def loop_scores_from_ripser(
    z: np.ndarray,
    max_points: int = 256,
    coeff: int = 2,
    normalize: bool = True,
    persistence_threshold: float = 0.0,
    prominence_ratio_threshold: float = 2.0,  # "2x" default
    rng=None,
):
    """
    Scores from H1 persistence:
      - max_persistence: strongest loop evidence
      - second_persistence: 2nd strongest (0 if none)
      - prominence_ratio: max/second (inf if second==0 and max>0, else 0)
      - sum_persistence: total H1 evidence
      - count_above_thresh: number of H1 features >= persistence_threshold
      - prominent: prominence_ratio >= prominence_ratio_threshold
    """
    if rng is None:
        rng = np.random.default_rng(0)

    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 4:
        return {
            "max_persistence": 0.0,
            "second_persistence": 0.0,
            "prominence_ratio": 0.0,
            "sum_persistence": 0.0,
            "count_above_thresh": 0,
            "prominent": False,
            "dgms_H1": np.zeros((0, 2), dtype=np.float64),
        }

    z = _subsample_time(z, max_points=max_points)

    out = ripser(z, maxdim=1, coeff=coeff)
    H1 = out["dgms"][1]
    if H1.size == 0:
        return {
            "max_persistence": 0.0,
            "second_persistence": 0.0,
            "prominence_ratio": 0.0,
            "sum_persistence": 0.0,
            "count_above_thresh": 0,
            "prominent": False,
            "dgms_H1": H1,
        }

    pers = H1[:, 1] - H1[:, 0]
    pers = pers[np.isfinite(pers)]
    if pers.size == 0:
        return {
            "max_persistence": 0.0,
            "second_persistence": 0.0,
            "prominence_ratio": 0.0,
            "sum_persistence": 0.0,
            "count_above_thresh": 0,
            "prominent": False,
            "dgms_H1": H1,
        }

    if normalize:
        scale = _traj_scale(z, rng=rng)
        pers = pers / scale

    pers_sorted = np.sort(pers)[::-1]  # descending
    max_p = float(pers_sorted[0])
    second_p = float(pers_sorted[1]) if pers_sorted.size > 1 else 0.0

    if second_p > 1e-12:
        pr = float(max_p / second_p)
    else:
        pr = float(np.inf) if max_p > 0 else 0.0

    sum_p = float(np.sum(pers_sorted))
    cnt = int(np.sum(pers_sorted >= persistence_threshold))
    prominent = bool(pr >= prominence_ratio_threshold)

    return {
        "max_persistence": max_p,
        "second_persistence": second_p,
        "prominence_ratio": pr,
        "sum_persistence": sum_p,
        "count_above_thresh": cnt,
        "prominent": prominent,
        "dgms_H1": H1,
    }


# ---- evaluation loop (minimal changes) ----
max_p_list = []
sum_p_list = []
cnt_list = []
pr_list = []
prominent_list = []

rng = np.random.default_rng(0)

PERSIST_THR = 0.2
PR_THR = 1.5

for sample in tqdm(test_ds):
    z = sample["z"]
    z = z - z.mean(axis=0)  # stabilize
    scores = loop_scores_from_ripser(
        z,
        max_points=256,
        coeff=2,
        normalize=True,
        persistence_threshold=PERSIST_THR,
        prominence_ratio_threshold=PR_THR,
        rng=rng,
    )
    max_p_list.append(scores["max_persistence"])
    sum_p_list.append(scores["sum_persistence"])
    cnt_list.append(scores["count_above_thresh"])
    pr_list.append(scores["prominence_ratio"])
    prominent_list.append(scores["prominent"])

max_p = np.asarray(max_p_list, dtype=np.float64)
sum_p = np.asarray(sum_p_list, dtype=np.float64)
cnt = np.asarray(cnt_list, dtype=np.int32)
pr = np.asarray(pr_list, dtype=np.float64)
prominent = np.asarray(prominent_list, dtype=bool)

def _stats(x):
    # keep it robust to infs from pr
    xf = x[np.isfinite(x)]
    return {
        "mean": float(np.mean(xf)) if xf.size else float("nan"),
        "median": float(np.median(xf)) if xf.size else float("nan"),
        "std": float(np.std(xf)) if xf.size else float("nan"),
        "p90": float(np.quantile(xf, 0.90)) if xf.size else float("nan"),
        "finite_frac": float(np.mean(np.isfinite(x))),
    }

print("Max H1 persistence (normalized):", _stats(max_p))
print("Sum H1 persistence (normalized):", _stats(sum_p))
print(f"Count >= {PERSIST_THR}:", _stats(cnt.astype(np.float64)))

print("\nProminence ratio = max/second:", _stats(pr))
print(f"  prominent (ratio >= {PR_THR}): {float(np.mean(prominent))}")