import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import detrend, savgol_filter, find_peaks
from Dataset import get_latents_dataset
from utils.topology import project_to_major_axis, von_mises_kernel_smoother
from tasks.Compute_EDES import EDES_via_Phase, find_peaks_sentinel

EDES_axis = np.load("tasks/EDES_axis.npy")

def global_axis(peaks, valleys):
    es_preds = peaks; ed_preds = valleys
    return ed_preds, es_preds

def time_interval_assignment(peaks, valleys):
    g1, g2 = (peaks, valleys) if peaks[0] < valleys[0] else (valleys, peaks)
    g1, g2 = np.asarray(g1), np.asarray(g2)
    n = min(len(g1), len(g2))
    g1, g2 = g1[:n], g2[:n]
    if n < 2:  # not enough points for mean intervals
        return (g1, g2) if (g2[0] - g1[0]) < (g1[1] - g2[0] if len(g1) > 1 else np.inf) else (g2, g1)
    m1 = (g2[:-1] - g1[:-1]).mean()
    m2 = (g1[1:] - g2[:-1]).mean()
    return (g1, g2) if m1 < m2 else (g2, g1)

def prewindow_velocity_assignment(z, peaks, valleys):
    dz_norms = np.linalg.norm(savgol_filter(z, deriv=1, window_length=11, polyorder=3, axis=0), axis=-1)
    def prewindow_mean(x, idxs, k=5):
        vals = []
        for i in np.asarray(idxs, dtype=int):
            start = max(0, i - k + 1)
            end = i+1
            vals.append(np.mean(x[start:end]))
        return np.mean(vals)
    peak_d = prewindow_mean(dz_norms, peaks, k=5)
    valley_d = prewindow_mean(dz_norms, valleys, k=5)
    if peak_d > valley_d:
        ed_preds = peaks; es_preds = valleys
    else:
        ed_preds = valleys; es_preds = peaks
    return ed_preds, es_preds

def acceleration_assignment(z, peaks, valleys):
    ddz_norms = np.linalg.norm(savgol_filter(z, deriv=2, window_length=11, polyorder=3, axis=0), axis=-1)
    peak_dd = np.mean(ddz_norms[peaks])
    valley_dd = np.mean(ddz_norms[valleys])
    if peak_dd > valley_dd:
        ed_preds = peaks; es_preds = valleys
    else:
        ed_preds = valleys; es_preds = peaks
    return ed_preds, es_preds

def phase_acceleration_assignment(phase, peaks, valleys):
    dphase = savgol_filter(np.unwrap(phase), deriv=2, window_length=11, polyorder=3, axis=0)
    peak_mu = np.mean(np.linalg.norm(dphase[peaks], axis=-1))
    valley_mu = np.mean(np.linalg.norm(dphase[valleys], axis=-1))
    if peak_mu > valley_mu:
        ed_preds = peaks; es_preds = valleys
    else:
        ed_preds = valleys; es_preds = peaks
    return ed_preds, es_preds

def speed_profile_assignment(z, peaks, valleys):
    n = z.shape[0]
    if peaks.size == 0 or valleys.size == 0:
        return peaks, valleys

    dz = savgol_filter(z, window_length=11, polyorder=3, deriv=1, axis=0)
    speed = np.linalg.norm(dz, axis=-1)

    def count_seg(a, b):
        if b - a < 5:
            return 0
        s = speed[a:b]
        pk, _ = find_peaks(s, prominence=0.3 * (s.max() - s.min()),
                           distance=5)
        return int(pk.size)

    pset, vset = set(peaks.tolist()), set(valleys.tolist())
    pset -= (pset & vset)
    vset -= (pset & vset)

    events = sorted([(i, 0) for i in pset if 0 <= i < n] + [(i, 1) for i in vset if 0 <= i < n])
    if len(events) < 2:
        return peaks, valleys

    p2v = v2p = 0
    for (i0, t0), (i1, t1) in zip(events[:-1], events[1:]):
        if t0 == 0 and t1 == 1:
            p2v += count_seg(i0, i1)
        elif t0 == 1 and t1 == 0:
            v2p += count_seg(i0, i1)

    return (valleys, peaks) if p2v > v2p else (peaks, valleys)

# def phase_density_assignment(z, phase, peaks, valleys):


def _worker(args):
    i, sample = args
    z = sample["z"]  # [T, latent_dim]
    phase = sample["phase"]  # [T]
    gt_ed = sample["ED"]
    gt_es = sample["ES"]
    fps = sample["metadata"]["FPS"]

    z = detrend(z, axis=0, type="linear")
    z_proj = savgol_filter(project_to_major_axis(z, phase, axis=EDES_axis), window_length=11, polyorder=3, axis=0)
    peaks, valleys = find_peaks_sentinel(z_proj, p=0.3, d=5)

    ed_preds, es_preds = global_axis(peaks, valleys)
    # ed_preds, es_preds = time_interval_assignment(peaks, valleys)
    # ed_preds, es_preds = prewindow_velocity_assignment(z,  peaks, valleys)
    # ed_preds, es_preds = acceleration_assignment(z, peaks, valleys)
    # ed_preds, es_preds = phase_acceleration_assignment(phase, peaks, valleys)
    # ed_preds, es_preds = voting_assignment(z, phase, peaks, valleys)
    # ed_preds, es_preds = speed_profile_assignment(z, peaks, valleys)

    ed_err = np.min(np.abs(ed_preds - gt_ed))
    es_err = np.min(np.abs(es_preds - gt_es))
    assign = (ed_err+es_err) <= (np.min(np.abs(es_preds-gt_ed))+np.min(np.abs(ed_preds-gt_es)))
    all_preds = np.concatenate([ed_preds, es_preds])
    min_err = [np.min(np.abs(all_preds - gt_ed)), np.min(np.abs(all_preds - gt_es))]
    return i, ed_err, es_err, assign, min_err


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_latents_dataset()

    ED_err_list = []
    ES_err_list = []
    assignments = []
    min_err_list = []
    large_error_idx = []

    # Prepare iterable of (index, sample) so we can recover i in results
    items = list(enumerate(test_ds))

    with ProcessPoolExecutor(max_workers=1) as ex:
        for i, ed_err, es_err, assign, min_err in tqdm(ex.map(_worker, items), total=len(items)):
            ED_err_list.append(ed_err)
            ES_err_list.append(es_err)
            assignments.append(assign)
            min_err_list.append(min_err)
            # if ed_err > 10 or es_err > 10:
            #     large_error_idx.append(i)

    ED_err_array = np.array(ED_err_list)
    ES_err_array = np.array(ES_err_list)

    print(f"Assignment accuracy: {np.mean(assignments)*100:.2f}%")
    print(f"Mean Localization error: {np.mean(min_err_list)} ± {np.std(min_err_list)}")

    print("ED error stats:")
    print(f"Range: [{ED_err_array.min():.4f}, {ED_err_array.max():.4f}]")
    print(f"Performance: {ED_err_array.mean():.4f} ± {ED_err_array.std():.4f} (mean ± std)")

    print("ES error stats:")
    print(f"Range: [{ES_err_array.min():.4f}, {ES_err_array.max():.4f}]")
    print(f"Performance: {ES_err_array.mean():.4f} ± {ES_err_array.std():.4f} (mean ± std)")

    # print(f"Large error indices: \n{large_error_idx}")