import subprocess
import sys

subprocess.run(
    ["torchrun", "--nproc_per_node=4", "EchoDynaDDP.py"],
    cwd="/workspace/miccai2026",
    check=True
)

subprocess.run(
    ["torchrun", "--nproc_per_node=4", "EchoDynaDDP1.py"],
    cwd="/workspace/miccai2026",
    check=True
)

subprocess.run(
    ["torchrun", "--nproc_per_node=4", "EchoDynaDDP2.py"],
    cwd="/workspace/miccai2026",
    check=True
)