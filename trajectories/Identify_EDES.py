import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from Dataset import get_latents_dataset
from tasks.Compute_EDES import EDES_via_Phase


def _worker(args):
    i, sample = args
    z = sample["z"]  # [T, latent_dim]
    gt_ed = sample["ED"]
    gt_es = sample["ES"]
    fps = sample["metadata"]["FPS"]

    ed_err, es_err = EDES_via_Phase(z, gt_ed=gt_ed, gt_es=gt_es)
    return i, ed_err, es_err


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_latents_dataset()

    ED_err_list = []
    ES_err_list = []
    large_error_idx = []

    # Prepare iterable of (index, sample) so we can recover i in results
    items = list(enumerate(test_ds))

    with ProcessPoolExecutor(max_workers=1) as ex:
        for i, ed_err, es_err in tqdm(ex.map(_worker, items), total=len(items)):
            ED_err_list.append(ed_err)
            ES_err_list.append(es_err)
            if ed_err > 10 or es_err > 10:
                large_error_idx.append(i)

    ED_err_array = np.array(ED_err_list)
    ES_err_array = np.array(ES_err_list)

    print("ED error stats:")
    print(f"Min: {ED_err_array.min():.4f}")
    print(f"Max: {ED_err_array.max():.4f}")
    print(f"Median: {np.median(ED_err_array):.4f}")
    print(f"Mean: {ED_err_array.mean():.4f}")
    print(f"Std: {ED_err_array.std():.4f}")

    print("ES error stats:")
    print(f"Min: {ES_err_array.min():.4f}")
    print(f"Max: {ES_err_array.max():.4f}")
    print(f"Median: {np.median(ES_err_array):.4f}")
    print(f"Mean: {ES_err_array.mean():.4f}")
    print(f"Std: {ES_err_array.std():.4f}")

    print(f"Large error indices: \n{large_error_idx}")