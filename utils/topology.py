import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt, savgol_filter

from dreimac import CircularCoords


def highpass_filter(signal, fs, cutoff=0.5, order=4, axis=0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered = filtfilt(b, a, signal, axis=axis)
    return filtered


def cohomology_circular_coords(
        z: np.ndarray, fps: int, 
        savgol: bool = False, highpass: bool = True, pca: bool = True,
        print_dgms_summary: bool = False):
    '''
    Extracts circular coordinates from the latent trajectory z using persistent cohomology.
    z: [T, D] numpy array
    fps: Frames per second of the signal
    savgol: Whether to apply Savitzky-Golay filter
    highpass: Whether to apply high-pass filter
    pca: Whether to apply PCA
    print_dgms_summary: Whether to print a summary of the persistence diagrams
    '''
    assert z.ndim == 2, "z should be a 2D array of shape [T, D]"

    if savgol:
        z = savgol_filter(z, window_length=11, polyorder=3, axis=0)
    if highpass:
        z = highpass_filter(z, fs=fps, cutoff=0.5, order=2, axis=0)
    if pca:
        pca_op = PCA(n_components=0.99)
        z = pca_op.fit_transform(z)

    cc = CircularCoords(z, n_landmarks=min(len(z), 1000))
    phase = cc.get_coordinates(perc=0.95)
    dgms = cc.dgms_

    if print_dgms_summary:
        if len(dgms) > 1:
            h1_dgm = dgms[1]
            persistences = h1_dgm[:, 1] - h1_dgm[:, 0]
            top_indices = np.argsort(persistences)[-5:][::-1]
            print("5 largest H1 persistences:")
            for index in top_indices:
                birth, death = h1_dgm[index]
                persistence = death - birth
                print(f"  [ {birth:.3f}, {death:.3f} ) - persistence: {persistence:.3f}")

    return phase, dgms


def plot_phase_and_z(z: np.ndarray, phase: np.ndarray, out_dir: str, index: int, gt_ed: int = None, frames_idx: list = None):
    if gt_ed is not None:
        ed_phase = phase[gt_ed]
        phase = (phase - ed_phase + 2*np.pi) % (2*np.pi)

    fig, ax = plt.subplots()
    scatter = ax.scatter(z[:,0], z[:,1], c=phase, s=6, cmap="viridis")
    if frames_idx is not None:
        for f in frames_idx:
            ax.scatter(z[f,0], z[f,1], color='red', marker="x", s=100, label="ES/ED", zorder=3)
    ax.set_title("Phase Color Plot of Z - PC1 & PC2")
    ax.axis("off")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Phase')
    cbar.ax.set_yticks([0, np.pi, 2*np.pi])
    cbar.ax.set_yticklabels(['0', 'π', '2π'])
    plt.savefig(os.path.join(out_dir, f"{index}-phase_color.png"), dpi=200)
    plt.clf()
    plt.close()


def plot_phase_and_time(phase: np.ndarray, timestamps: np.ndarray, out_dir: str, 
                        index: int, frames_idx: list = None, sine: bool = True,
                        differentiate: int = 0):
    if sine:
        phase = np.sin(phase)

    for _ in range(differentiate):
        phase = np.gradient(phase)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(timestamps, phase, color='blue', s=3)
    if frames_idx is not None:
        for f in frames_idx:
            ax.scatter(timestamps[f], phase[f], color='red', marker="x", s=100, label="ES/ED", zorder=3)
    phase_vals = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi] if sine else [-1, 0, 1]
    for phase_val in phase_vals:
        ax.axhline(phase_val, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Time (s)")
    if sine:
        ax.set_ylabel("Sin Phase")
    else:
        ax.set_ylabel("Phase")
        ax.set_yticks([0, np.pi, 2*np.pi])
        ax.set_yticklabels(['0', 'π', '2π'])
    plt.savefig(os.path.join(out_dir, f"{index}-phase_time-{differentiate}xdiff.png"), dpi=200)
    plt.clf()
    plt.close()


def plot_znorm_and_time(z: np.ndarray, timestamps: np.ndarray, out_dir: str, index: int, frames_idx: list = None):
    z_motion_norm = np.linalg.norm(z.clone().detach().cpu().numpy(), axis=1).squeeze()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, z_motion_norm, color='green', linewidth=1)
    if frames_idx is not None:
        for f in frames_idx:
            ax.scatter(timestamps[f], z_motion_norm[f], color='red', marker="x", s=100, label="ES/ED", zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Norm of z_motion")
    plt.savefig(os.path.join(out_dir, f"{index}-znorm_time.png"), dpi=200)
    plt.clf()
    plt.close()