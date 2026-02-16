import subprocess
import sys

scripts = [
    "tasks/EF_VViT.py",
    "tasks/Seg_VViT.py",
]

for script in scripts:
    subprocess.run(
        [sys.executable, script],
        cwd="/workspace/miccai2026",
        check=True,   # raises CalledProcessError if it fails
    )