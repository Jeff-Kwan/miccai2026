import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import butter, filtfilt


def highpass_filter(signal, fs, cutoff=0.5, order=4, axis=0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered = filtfilt(b, a, signal, axis=axis)
    return filtered


def detect_baseline_wander(signal_data, sampling_rate, cutoff_freq=0.5):
    fft = np.fft.fft(signal_data)
    frequencies = np.fft.fftfreq(len(signal_data), 1 / sampling_rate)

    low_freq_mask = np.abs(frequencies) < cutoff_freq
    low_freq_power = np.sum(np.abs(fft[low_freq_mask]) ** 2)
    total_power = np.sum(np.abs(fft) ** 2)

    ratio = low_freq_power / total_power
    return ratio > 0.1  # threshold can be adjusted


def compute_main_orientation_and_extrema(
    trajectory,
    fps,
    theta_threshold_degrees=30,
    ransac_iterations=100,
    window_length=8,
    polyorder=2,
    edge_events=True,
):
    """
    Generalized to N-D trajectory input: trajectory shape (T, N).

    Returns:
    - group1: indices of points belonging to the first group.
    - group2: indices of points belonging to the second group.
    - endpoint1: position of one end point of the main axis (shape (N,))
    - endpoint2: position of the second end point of the main axis (shape (N,))
    - trajectory_projected: 1D trajectory after projecting onto the main axis (shape (T,))
    - direction: main axis unit vector (shape (N,))
    """
    # --- Step 1: Compute main orientation using RANSAC & PCA ---
    displacements = np.diff(trajectory, axis=0)  # (T-1, N)
    magnitudes = np.linalg.norm(displacements, axis=1)
    nonzero_mask = magnitudes > 1e-8
    displacements = displacements[nonzero_mask]
    magnitudes = magnitudes[nonzero_mask]

    if len(displacements) == 0:
        raise ValueError("All displacements are zero.")

    unit_vectors = displacements / magnitudes[:, np.newaxis]
    theta_threshold = np.deg2rad(theta_threshold_degrees)
    cos_threshold = np.cos(theta_threshold)

    best_num_inliers = -1
    best_inliers = None

    for _ in range(ransac_iterations):
        idx = np.random.randint(len(unit_vectors))
        hypothesis = unit_vectors[idx]
        dots = np.abs(np.dot(unit_vectors, hypothesis))
        inliers = dots >= cos_threshold
        num_inliers = np.sum(inliers)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers

    if best_num_inliers == 0:
        raise ValueError("No inliers found.")

    inlier_displacements = displacements[best_inliers]
    pca = PCA(n_components=1)
    pca.fit(inlier_displacements)
    main_orientation = pca.components_[0]

    # use unit direction for projection + endpoints
    direction = main_orientation / np.linalg.norm(main_orientation)

    # --- Step 2: Project trajectory onto main axis to find extrema ---
    mean_point = np.mean(trajectory, axis=0)
    trajectory_projected = np.dot(trajectory - mean_point, direction)

    t_min, t_max = np.min(trajectory_projected), np.max(trajectory_projected)
    endpoint1 = mean_point + t_min * direction
    endpoint2 = mean_point + t_max * direction

    # --- Step 3: Detect peaks and valleys in the projections ---
    smoothed_proj = savgol_filter(trajectory_projected, window_length, polyorder)

    wander_flag = detect_baseline_wander(trajectory_projected, fps)
    if wander_flag:
        new_traj = np.pad(smoothed_proj, (10, 10), "reflect")
        filtered_proj = highpass_filter(new_traj, fps)[10:-10]
    else:
        filtered_proj = smoothed_proj
    filtered_proj = trajectory_projected

    prominence_threshold = 0.3 * (np.max(filtered_proj) - np.min(filtered_proj))

    if edge_events:
        peak_input = np.concatenate(([min(filtered_proj)], filtered_proj, [min(filtered_proj)]))
        valley_input = np.concatenate(([max(filtered_proj)], filtered_proj, [max(filtered_proj)]))
    else:
        peak_input = filtered_proj
        valley_input = filtered_proj

    peaks, _ = find_peaks(peak_input, prominence=prominence_threshold)
    valleys, _ = find_peaks(-valley_input, prominence=prominence_threshold)

    if edge_events:
        peaks = peaks - 1
        valleys = valleys - 1

    # --- Step 4: Group direction changes (same convention as original) ---
    group1 = np.array(valleys)  # ED
    group2 = np.array(peaks)    # ES

    return group1, group2, endpoint1, endpoint2, trajectory_projected, direction