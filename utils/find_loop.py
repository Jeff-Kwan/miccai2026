import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

_TWOPI = 2.0 * np.pi


# -------------------------------
# Core circular utilities
# -------------------------------

def _wrap01(theta01):
    th = np.asarray(theta01, dtype=np.float64)
    return th - np.floor(th)


def _circ_to_unit(theta01):
    ang = _TWOPI * _wrap01(theta01)
    return np.cos(ang), np.sin(ang)


def _unit_to_circ01(c, s):
    return (np.arctan2(s, c) / _TWOPI) % 1.0


def _circular_diff_turns(theta01):
    th = _wrap01(theta01)
    d = th[1:] - th[:-1]
    return (d + 0.5) % 1.0 - 0.5


def _valid_savgol_window(n, window_length, polyorder):
    if n < 3:
        return 0
    wl = int(window_length)
    if wl < 3:
        wl = 3
    if wl % 2 == 0:
        wl += 1
    wl = max(wl, int(polyorder) + 3)
    if wl > n:
        wl = n if (n % 2 == 1) else (n - 1)
    return wl if wl >= int(polyorder) + 2 else 0


# -------------------------------
# Public API (same function names)
# -------------------------------

def smooth_theta_on_circle(theta01, window_length=9, polyorder=2):
    """
    Wrap-safe smoothing:
      - convert theta -> (cos,sin)
      - SavGol smooth
      - renormalize back to unit circle
      - convert back to theta in [0,1)
    """
    th = _wrap01(theta01)
    n = th.size
    wl = _valid_savgol_window(n, window_length, polyorder)
    if wl == 0:
        return th.copy()

    c, s = _circ_to_unit(th)
    c = savgol_filter(c, wl, polyorder, mode="interp")
    s = savgol_filter(s, wl, polyorder, mode="interp")

    r = np.hypot(c, s)
    r = np.maximum(r, 1e-12)
    c /= r
    s /= r
    return _unit_to_circ01(c, s)


def unwrap_theta_turns(theta01):
    """
    Unwrap theta into continuous turns using minimal circular diffs.
    """
    th = _wrap01(theta01)
    out = np.empty(th.size, dtype=np.float64)
    out[0] = th[0]
    if th.size > 1:
        out[1:] = out[0] + np.cumsum(_circular_diff_turns(th))
    return out


def segment_periods(theta_turns, min_points_per_period=10, min_coverage_turns=0.80):
    """
    Segment cycles via integer turn crossings.
    Keeps segments that have enough points and phase coverage.
    Returns:
      cycle_ids: floor(theta_turns) for each sample
      periods: list of index arrays for each kept period
      boundaries: start indices of kept periods
    """
    tt = np.asarray(theta_turns, dtype=np.float64)
    T = tt.size
    if T == 0:
        return np.zeros(0, dtype=int), [], np.zeros(0, dtype=int)

    cycle_ids = np.floor(tt).astype(int)
    changes = np.flatnonzero(cycle_ids[1:] != cycle_ids[:-1]) + 1
    bounds = np.unique(np.concatenate(([0], changes, [T])))

    periods = []
    boundaries = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        a = int(a)
        b = int(b)
        if b - a < int(min_points_per_period):
            continue
        seg = tt[a:b]
        if (seg.max() - seg.min()) < float(min_coverage_turns):
            continue
        periods.append(np.arange(a, b, dtype=int))
        boundaries.append(a)

    return cycle_ids, periods, np.asarray(boundaries, dtype=int)


def compute_circular_coordinate_largest_h1(X, maxdim=2, n_landmarks=200, smooth_window_length=9, smooth_polyorder=2,
                                          min_points_per_period=10, min_coverage_turns=0.80, verbose=True):
    """
    Minimal, self-contained pipeline:
      1) PH via ripser (returns dgms)
      2) circular coordinate via dreimac CircularCoords (theta in [0,1))
      3) wrap-safe smoothing on circle
      4) unwrap to continuous turns
      5) segment periods by integer crossings

    Returns:
      dgms, theta01_smooth, theta_turns, cycle_ids, periods, diagnostics
    """
    from ripser import ripser
    from dreimac import CircularCoords

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be (T, d)")

    dgms = ripser(X, maxdim=maxdim)["dgms"]

    cc = CircularCoords(X, n_landmarks=min(int(n_landmarks), len(X)))
    theta01 = cc.get_coordinates(perc=0.95, cocycle_idx=0)
    theta01_s = smooth_theta_on_circle(theta01, window_length=smooth_window_length, polyorder=smooth_polyorder)
    theta_turns = unwrap_theta_turns(theta01_s)

    cycle_ids, periods, boundaries = segment_periods(
        theta_turns,
        min_points_per_period=min_points_per_period,
        min_coverage_turns=min_coverage_turns,
    )

    if verbose:
        print("[phase] periods kept:", len(periods))

    diagnostics = dict(
        theta01_raw=_wrap01(theta01),
        theta01_smooth=theta01_s,
        boundaries=boundaries,
    )

    return dgms, theta01_s, theta_turns, cycle_ids, periods, diagnostics


def plot_theta_and_cycles(theta01, theta_turns, cycle_ids, out_prefix="ph_phase"):
    t = np.arange(len(theta01))

    plt.figure(figsize=(10, 3))
    plt.plot(t, theta01)
    plt.title("Circular coordinate θ (mod 1)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_theta_mod1.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(t, theta_turns)
    plt.title("Unwrapped phase (turns)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_theta_unwrapped.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(t, cycle_ids)
    plt.title("Cycle id = floor(unwrapped turns)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cycle_ids.png", dpi=200)
    plt.close()


def loop_polyline_for_period(X, idx, theta01, nbins=64):
    """
    Build a representative loop polyline for one period:
      - sort by theta
      - average points in theta bins
      - close the loop
    """
    Xp = np.asarray(X, dtype=np.float64)[idx]
    th = _wrap01(np.asarray(theta01, dtype=np.float64)[idx])

    order = np.argsort(th)
    Xp = Xp[order]
    th = th[order]

    bins = np.linspace(0.0, 1.0, int(nbins) + 1)
    pts = []
    for j in range(len(bins) - 1):
        m = (th >= bins[j]) & (th < bins[j + 1])
        if np.any(m):
            pts.append(Xp[m].mean(axis=0))
    pts = np.asarray(pts, dtype=np.float64)

    if pts.shape[0] > 1:
        pts = np.vstack([pts, pts[0]])
    return pts


def plot_pointcloud_with_period_loops(X, periods, theta01, out_png="period_loops.png", max_periods_to_draw=3):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 2:
        print("X has <2 dims, skipping 2D plot.")
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], s=10)

    for idx in periods[: int(max_periods_to_draw)]:
        poly = loop_polyline_for_period(X, idx, theta01, nbins=64)
        if poly.shape[0] > 1:
            plt.plot(poly[:, 0], poly[:, 1], linewidth=2)

    plt.title(f"Point cloud + period loop polylines (first {min(int(max_periods_to_draw), len(periods))})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()




# -------------------------
# Example
# -------------------------
if __name__ == "__main__":
    # Demo synthetic: two laps around a noisy circle
    t = np.linspace(0, 4*np.pi, 600)  # 2 turns
    X = np.c_[np.cos(t), np.sin(t)] + 0.05*np.random.randn(len(t), 2)