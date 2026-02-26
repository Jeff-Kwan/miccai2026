from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from Dataset import get_latents_dataset

train_ds, val_ds, test_ds = get_latents_dataset()


monotonicity_scores = []
for sample in tqdm(test_ds):
    phase = sample["phase"]  # [T]
    dphase = np.diff(np.unwrap(phase))
    dphase_direction = dphase >= 0
    score = np.mean(dphase_direction)
    monotonicity_scores.append(score)

print(f"Mean monotonicity score: {np.mean(monotonicity_scores):.3f} ± {np.std(monotonicity_scores):.3f}")
print(f"Max score: {np.max(monotonicity_scores):.3f}, Min score: {np.min(monotonicity_scores):.3f}")
print(f"Percentage of samples with score > 0.9: {np.mean(np.array(monotonicity_scores) > 0.9) * 100:.2f}%")

# plot histogram
plt.hist(monotonicity_scores, bins=20, edgecolor='black')
plt.title("Histogram of Phase Monotonicity Scores")
plt.xlabel("Monotonicity Score")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.savefig("phase_monotonicity_histogram.png")