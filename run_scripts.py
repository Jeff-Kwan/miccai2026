import subprocess
import sys

scripts = [
    "TrainSAE.py",
    "tasks/Compute_EDES.py",
]

for script in scripts:
    subprocess.run(
        [sys.executable, script],
        cwd="/workspace/miccai2026",
        check=True
    )