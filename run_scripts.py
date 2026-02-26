import subprocess

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

subprocess.run(
    ["torchrun", "--nproc_per_node=4", "EchoDynaDDP3.py"],
    cwd="/workspace/miccai2026",
    check=True
)

subprocess.run(
    ["torchrun", "--nproc_per_node=4", "EchoDynaDDP4.py"],
    cwd="/workspace/miccai2026",
    check=True
)