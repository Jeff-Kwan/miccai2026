import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt, detrend, savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import LocalOutlierFactor
from dreimac import CircularCoords
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, diags, csgraph, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter1d


#####
# Utility
####

def highpass_filter(signal, fs, cutoff=0.5, order=4, axis=0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered = filtfilt(b, a, signal, axis=axis)
    return filtered

def robust_z(z):
    # Dimensionality Reduction
    pca = PCA(n_components=0.95)
    z_reduced = pca.fit_transform(z)

    # LoF Robust Inliers
    lof = LocalOutlierFactor(n_neighbors=15)
    _ = lof.fit_predict(z_reduced)
    lof_score = lof.negative_outlier_factor_
    mask = lof_score > np.quantile(lof_score, 0.02)
    fit_idx  = np.flatnonzero(mask)
    hold_idx = np.flatnonzero(~mask)
    # print(f"LOF: Keeping {len(fit_idx)} inliers, holding out {len(hold_idx)} points")
    return fit_idx, hold_idx


def preprocess_z(z):
    z = detrend(z, axis=0, type='linear')
    z = savgol_filter(z, deriv=1, window_length=11, polyorder=3, axis=0)
    z = detrend(z, axis=0, type='linear')
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    return z

#####
# Topological
#####

def laplacian_phase(
    X: np.ndarray,
    k: int = 15,
    eps: float = 1e-12,
    n_eigs: int = 6,
):
    """Graph-Laplacian (diffusion-map style) phase for a quasi-1D loop manifold."""
    if X.ndim != 2:
        raise ValueError("X must be [T, D]")
    T = X.shape[0]
    if k < 2:
        raise ValueError("k must be >= 2")
    if T < max(10, k + 2):
        raise ValueError(f"Need more points than kNN. Got T={T}, k={k}.")
    if n_eigs < 4:
        raise ValueError("n_eigs should be >= 4 for reliable pair selection")

    # ---- preprocessing ----
    X = preprocess_z(X)

    # ---- kNN graph (directed) ----
    dists, idx = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X).kneighbors(X)
    dists, idx = dists[:, 1:], idx[:, 1:]  # drop self
    cos_sim = 1.0 - dists   # [1.0, -1.0]
    w_knn = cos_sim.clip(min=0.0) ** 2.0

    # ---- Mutual & Symmetric ----
    rows = np.repeat(np.arange(T), k)
    cols = idx.reshape(-1)
    dat  = w_knn.reshape(-1)

    # weighted directed adjacency
    W = coo_matrix((dat, (rows, cols)), shape=(T, T)).tocsr()

    # keep only mutual edges: (i->j) AND (j->i)
    mutual = W.multiply(W.T) > 0        # nonzero only where both directions exist
    W = W.multiply(mutual)              # keep weights only on mutual edges

    # symmetrize
    W = (W + W.T) * 0.5

    # Lazy random walk
    W = W + diags(np.full(T, 1e-4), format="csr")

    deg = np.asarray(W.sum(axis=1)).ravel()
    if np.any(deg <= eps):
        raise ValueError("Graph has near-zero degree nodes")
    W.eliminate_zeros()

    # ---- normalized Laplacian ----
    L = csgraph.laplacian(W, normed=True)

    # ---- eigensolve ----
    m = min(int(n_eigs), T - 2)
    try:
        evals, evecs = eigsh(L, k=m, which="SA")
    except Exception:
        evals, evecs = eigsh(L, k=m, sigma=0.0, which="LM")

    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]

    # ---- choose eigenvector pair ----
    # upper = min(m, 8)
    # best_score, best_pair = -np.inf, None
    # for i in range(1, upper):
    #     a = evecs[:, i]
    #     for j in range(i + 1, upper):
    #         b = evecs[:, j]
    #         r = np.hypot(a, b) + eps
    #         r_cv = r.std() / (r.mean() + eps)
    #         spread = a.std() + b.std()
    #         lam_gap = abs(evals[j] - evals[i]) / (abs(evals[i]) + abs(evals[j]) + eps)
    #         score = -r_cv + 0.1 * spread - 0.1 * lam_gap
    #         if score > best_score:
    #             best_score, best_pair = score, (i, j)

    # if best_pair is None:
    #     raise RuntimeError("Could not find a suitable eigenvector pair for phase.")

    # i, j = best_pair
    i, j = 1, 2
    theta = np.arctan2(evecs[:, j], evecs[:, i])

    # --- enforce increasing phase over time (sign convention) ---
    if np.nanmean(np.diff(np.unwrap(theta))) < 0:
        theta = -theta              # flip orientation

    return theta, evals, evecs



def cohomology_circular_coords(
        z: np.ndarray, alpha=0.8,
        print_dgms_summary: bool = False):
    '''
    Extracts circular coordinates from the latent trajectory z using persistent cohomology.
    z: [T, D] numpy array
    fps: Frames per second of the signal
    print_dgms_summary: Whether to print a summary of the persistence diagrams
    '''
    assert z.ndim == 2, "z should be a 2D array of shape [T, D]"

    # Soft Linear detrend
    # z = z - z.mean(axis=0, keepdims=True)
    # z = alpha*detrend(z, axis=0, type='linear') + (1-alpha)*z
    z = detrend(z, axis=0, type='linear')

    # Robustly identify inlier points for circular coordinate fitting
    fit_idx, hold_idx = robust_z(z)

    # Soft normalize
    # norms = np.linalg.norm(z, axis=-1, keepdims=True)
    # norms = alpha*np.median(norms, axis=0, keepdims=True) + (1-alpha)*norms
    # z = z / norms
    
    z_fit = z[fit_idx]
    z_hold = z[hold_idx] if len(hold_idx) > 0 else None

    pca = PCA(n_components=0.95)
    z_fit = pca.fit_transform(z_fit)
    z_hold = pca.transform(z_hold)

    # Build rectangular distance matrix:
    if len(hold_idx) == 0:
        D_fit_fit = np.linalg.norm(z_fit[:, None, :] - z_fit[None, :, :], axis=-1)  # (N_fit, N_fit)
        D_rect = D_fit_fit
    else:
        D_fit_fit = np.linalg.norm(z_fit[:, None, :] - z_fit[None, :, :], axis=-1)      # (N_fit, N_fit)
        D_fit_hold = np.linalg.norm(z_fit[:, None, :] - z_hold[None, :, :], axis=-1)    # (N_fit, N_hold)
        D_rect = np.hstack([D_fit_fit, D_fit_hold])                                     # (N_fit, N_all)
    
    # dz = preprocess_z(z)
    # D_rect2 = (2.0 - dz @ dz.T)**2

    # z = detrend(z, axis=0, type='linear')
    # z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    # D_rect1 = (2.0 - z @ z.T)**2

    # D_rect = D_rect1 + D_rect2

    # Fit circular coords using only fit subset, but assign to all columns
    cc = CircularCoords(D_rect, n_landmarks=len(z), distance_matrix=True)
    phase = cc.get_coordinates()
    dgms = cc.dgms_

    if print_dgms_summary:
        if len(dgms) > 1:
            h1_dgm = dgms[1]
            persistences = h1_dgm[:, 1] - h1_dgm[:, 0]
            top_indices = np.argsort(persistences)[-3:][::-1]
            print("3 largest H1 persistences:")
            for index in top_indices:
                birth, death = h1_dgm[index]
                persistence = death - birth
                print(f"  [ {birth:.3f}, {death:.3f} ) - persistence: {persistence:.3f}")

    return phase, dgms


def project_to_phase_plane(z: np.ndarray, phase: np.ndarray):
    z = preprocess_z(z)
    pls = PLSRegression(n_components=2)
    pls.fit(np.column_stack([np.cos(phase), np.sin(phase)]), z)
    C = pls.coef_  # mapping from X (2) -> z (D): shape usually (2, D)
    if C.shape[0] == 2:
        C = C.T  # (D, 2)
    # 4) Build an orthonormal basis for the "phase plane" in z-space
    # Columns of U span the column-space of C
    U, _, _ = np.linalg.svd(C, full_matrices=False)
    # Optional: make sign deterministic like your major-axis sign fix
    U[:, 0] *= np.sign(U[0, 0]) if U[0, 0] != 0 else 1.0
    U[:, 1] *= np.sign(U[0, 1]) if U[0, 1] != 0 else 1.0
    # 5) Project z to basis
    z_proj = z @ U  # (N, D) @ (D, 2) -> (N, 2)
    return z_proj, U


def find_phase_major_axis(z: np.ndarray, phase: np.ndarray):
    z = preprocess_z(z)
    pls = PLSRegression(n_components=2)
    pls.fit(np.column_stack([np.cos(phase), np.sin(phase)]), z)
    C = pls.coef_
    if C.shape[0] == 2:
        C = C.T  # (D, 2)
    U, _, _ = np.linalg.svd(C, full_matrices=False)
    u0 = U[:, 0]     # major-axis direction (unit)
    u0 = u0 * np.sign(u0[0])  # fix sign ambiguity
    return u0



def von_mises_kernel_smoother(
    z: np.ndarray,
    phase: np.ndarray,
    kappa: float,
    grid: np.ndarray | None = None,
    n_grid: int = 256,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kernel smoother on the circle using a von Mises kernel:
        K(Δ) = exp(kappa * cos(Δ))

    Parameters
    ----------
    z : (T, D) array
        High-dimensional points.
    phase : (T,) array
        Phase labels in radians. Can be any real values; treated mod 2π.
    kappa : float
        Concentration parameter. Larger => narrower kernel.
    grid : (G,) array, optional
        Phases at which to evaluate the smoothed mean cycle (radians).
        If None, uses an evenly spaced grid over [-pi, pi).
    n_grid : int
        Number of grid points if grid is None.
    eps : float
        Small constant to avoid divide-by-zero.

    Returns
    -------
    grid : (G,) array
        Phase grid in radians ([-pi, pi) if generated).
    mu : (G, D) array
        Smoothed mean cycle evaluated at each grid phase.
    """
    z = np.asarray(z, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)

    if grid is None:
        grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False, dtype=np.float64)
    else:
        grid = np.asarray(grid, dtype=np.float64).ravel()

    # Pairwise circular differences: Δ_{g,t} = grid[g] - phase[t]
    # cos(Δ) handles periodicity automatically.
    delta = grid[:, None] - phase[None, :]
    w = np.exp(kappa * np.cos(delta))  # (G, T)

    # Normalize weights per grid point
    w_sum = np.sum(w, axis=1, keepdims=True)  # (G, 1)
    w_sum = np.maximum(w_sum, eps)

    # Weighted average: (G, T) @ (T, D) -> (G, D)
    mu = (w @ z) / w_sum

    return grid, mu


#####
# Plotting
#####



def plot_phase_and_z(z: np.ndarray, phase: np.ndarray, out_dir: str, index: int, gt_ed: int = None, frames_idx: list = None,
                     dim: str = "3d", mu: np.ndarray = None):
    if gt_ed is not None:
        ed_phase = phase[gt_ed]
        phase = (phase - ed_phase + 2*np.pi) % (2*np.pi)

    fig = plt.figure(figsize=(10, 8))
    
    if dim == "2d":
        ax = fig.add_subplot(111)
        scatter = ax.scatter(z[:,0], z[:,1], c=phase, s=6, cmap="viridis")
        if frames_idx is not None:
            for f in frames_idx:
                ax.scatter(z[f,0], z[f,1], color='red', marker="x", s=100, label="ES/ED", zorder=3)
        if mu is not None:
            ax.plot(mu[:,0], mu[:,1], color='black', linewidth=2, label='Mean cycle')
        ax.set_title("Phase Color Plot of Z - PC1 & PC2")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    elif dim == "3d":
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(z[:,0], z[:,1], z[:,2], c=phase, s=10, cmap="viridis")
        if frames_idx is not None:
            for f in frames_idx:
                ax.scatter(z[f,0], z[f,1], z[f,2], color='red', marker="x", s=100, label="ES/ED", zorder=3)
        if mu is not None:
            ax.plot(mu[:,0], mu[:,1], mu[:,2], color='black', linewidth=2, label='Mean cycle')
        ax.set_title("Phase Color Plot of Z - PC1, PC2 & PC3")
    else:
        raise ValueError("dim should be '2d' or '3d'")
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Phase')
    cbar.ax.set_yticks([0, np.pi, 2*np.pi])
    cbar.ax.set_yticklabels(['0', 'π', '2π'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index}-phase_color_{dim}.png"), dpi=200)
    plt.clf()
    plt.close()


def plot_phase_and_time(phase: np.ndarray, timestamps: np.ndarray, out_dir: str, 
                        index: int, frames_idx: list = None, gt_ed: int = None,
                        sine: bool = True, differentiate: int = 0):
    if gt_ed is not None:
        ed_phase = phase[gt_ed]
        phase = (phase - ed_phase + 2*np.pi) % (2*np.pi)

    if sine:
        phase = np.sin(phase)

    for _ in range(differentiate):
        phase = np.gradient(phase)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(timestamps, phase, color='blue', s=3)
    if frames_idx is not None:
        plt.axvline(x=timestamps[frames_idx[1]], color='blue', linestyle='--', label='Ground Truth ED')
        plt.axvline(x=timestamps[frames_idx[0]], color='green', linestyle='--', label='Ground Truth ES')
    phase_vals = [-1, 0, 1] if sine else [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    for phase_val in phase_vals:
        ax.axhline(phase_val, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Time (s)")
    if sine:
        ax.set_ylabel("Sin Phase")
        ax.set_yticks([-1, 0, 1])
    else:
        ax.set_ylabel("Phase")
        ax.set_yticks([0, np.pi, 2*np.pi])
        ax.set_yticklabels(['0', 'π', '2π'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index}-phase_time-{differentiate}xdiff.png"), dpi=200)
    plt.clf()
    plt.close()


def plot_znorm_and_time(z: np.ndarray, timestamps: np.ndarray, out_dir: str, index: int, frames_idx: list = None):
    z_norm = np.linalg.norm(z, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, z_norm, color='green', linewidth=1)
    if frames_idx is not None:
        plt.axvline(x=timestamps[frames_idx[1]], color='blue', linestyle='--', label='Ground Truth ED')
        plt.axvline(x=timestamps[frames_idx[0]], color='green', linestyle='--', label='Ground Truth ES')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Norm of z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index}-znorm_time.png"), dpi=200)
    plt.clf()
    plt.close()


def plot_phase_major_axis(z: np.ndarray, timestamps: np.ndarray, phase: np.ndarray, out_dir: str, index: int, 
                          frames_idx: list = None, peaks: list = None):
    major_axis = find_phase_major_axis(z, phase)
    z_proj = z @ major_axis
    plt.scatter(timestamps, z_proj, c=phase, cmap='hsv', s=5)
    cbar = plt.colorbar(label='Circular Coordinate Phase')
    cbar.ax.set_yticks([0, np.pi, 2*np.pi])
    cbar.ax.set_yticklabels(['0', 'π', '2π'])
    plt.xlabel('Time (s)')
    plt.ylabel('Projection on Major Axis')
    if frames_idx is not None:
        plt.axvline(x=timestamps[frames_idx[1]], color='blue', linestyle='--', label='Ground Truth ED')
        plt.axvline(x=timestamps[frames_idx[0]], color='green', linestyle='--', label='Ground Truth ES')
    if peaks is not None:
        for p in peaks:
            plt.scatter(timestamps[p], z_proj[p], color='black', marker="x", s=50, label="Peaks", zorder=3)
    plt.title('1D Projection of Latent Trajectory Colored by Phase')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index}-phase_color_1d.png"), dpi=200)
    plt.clf()
    plt.close()