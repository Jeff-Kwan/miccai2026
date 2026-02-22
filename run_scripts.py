import subprocess
import sys

subprocess.run(
    ["torchrun", "--nproc_per_node=4", "EchoDynaDDP.py"],
    cwd="/workspace/miccai2026",
    check=True
)