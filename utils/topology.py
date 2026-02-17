import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.cross_decomposition import PLSRegression
from dreimac import CircularCoords

#####
# Utility
####

def highpass_filter(signal, fs, cutoff=0.5, order=4, axis=0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered = filtfilt(b, a, signal, axis=axis)
    return filtered


#####
# Topological
#####

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
        pca_op = PCA(n_components=0.95)
        z = pca_op.fit_transform(z)

    cc = CircularCoords(z, n_landmarks=min(len(z), 1000))
    try:
        phase = cc.get_coordinates(perc=0.95)
    except Exception as e:
        print(f"Error in computing circular coordinates: {e}")
        phase = cc.get_coordinates(perc=0.95, standard_range=False)
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


def project_to_principal_plane(z: np.ndarray, phase: np.ndarray):
    '''
    Projects the latent trajectory z onto the 2D plane that best captures the circular coordinate phase using PLS regression.
    z: [T, D] numpy array
    phase: [T,] numpy array of circular coordinates (in radians)
    '''
    pls = PLSRegression(n_components=2)
    pls.fit(z, np.column_stack((np.sin(phase), np.cos(phase))))
    z_2d = pls.transform(z)
    return z_2d


def find_phase_major_axis(z: np.ndarray, phase: np.ndarray):
    Y = np.column_stack([np.sin(phase), np.cos(phase)])  # (N,2)
    pls = PLSRegression(n_components=2)
    pls.fit(Y, z)
    C = pls.coef_
    if C.shape[0] == 2:
        C = C.T  # (D, 2)
    U, _, _ = np.linalg.svd(C, full_matrices=False)
    u0 = U[:, 0]     # major-axis direction (unit)
    u0 = u0 * np.sign(u0[0])  # fix sign ambiguity
    return u0


#####
# Plotting
#####



def plot_phase_and_z(z: np.ndarray, phase: np.ndarray, out_dir: str, index: int, gt_ed: int = None, frames_idx: list = None,
                     dim: str = "3d"):
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
        ax.set_title("Phase Color Plot of Z - PC1 & PC2")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    elif dim == "3d":
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(z[:,0], z[:,1], z[:,2], c=phase, s=10, cmap="viridis")
        if frames_idx is not None:
            for f in frames_idx:
                ax.scatter(z[f,0], z[f,1], z[f,2], color='red', marker="x", s=100, label="ES/ED", zorder=3)
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
        for f in frames_idx:
            ax.scatter(timestamps[f], phase[f], color='red', marker="x", s=100, label="ES/ED", zorder=3)
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
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index}-phase_time-{differentiate}xdiff.png"), dpi=200)
    plt.clf()
    plt.close()


def plot_znorm_and_time(z: np.ndarray, timestamps: np.ndarray, out_dir: str, index: int, frames_idx: list = None):
    z_norm = np.linalg.norm(z, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, z_norm, color='green', linewidth=1)
    if frames_idx is not None:
        for f in frames_idx:
            ax.scatter(timestamps[f], z_norm[f], color='red', marker="x", s=100, label="ES/ED", zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Norm of z")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index}-znorm_time.png"), dpi=200)
    plt.clf()
    plt.close()


def plot_phase_major_axis(z: np.ndarray, timestamps: np.ndarray, phase: np.ndarray, out_dir: str, index: int, frames_idx: list = None):
    major_axis = find_phase_major_axis(z, phase)
    z_proj = z @ major_axis
    plt.scatter(timestamps, z_proj, c=phase, cmap='hsv', s=5)
    cbar = plt.colorbar(label='Circular Coordinate Phase')
    cbar.ax.set_yticks([0, np.pi, 2*np.pi])
    cbar.ax.set_yticklabels(['0', 'π', '2π'])
    plt.xlabel('Time (s)')
    plt.ylabel('Projection on Major Axis')
    if frames_idx is not None:
        for f in frames_idx:
            plt.scatter(timestamps[f], z_proj[f], color='red', marker="x", s=100, label="ES/ED", zorder=3)
    plt.title('1D Projection of Latent Trajectory Colored by Phase')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index}-phase_color_1d.png"), dpi=200)
    plt.clf()
    plt.close()