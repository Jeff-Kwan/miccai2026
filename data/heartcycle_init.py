import subprocess

# Install AWS CLI
setup_commands = [
    'curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"',
    'unzip awscliv2.zip',
    'sudo ./aws/install',
    'aws --version',
]

for cmd in setup_commands:
    subprocess.run(cmd, shell=True, check=True)

# Sync heartcycle data
cmd = [
    "aws",
    "s3",
    "sync",
    "--no-sign-request",
    "s3://physionet-open/heartcycle/1.0.0/",
    "data/heartcycle/",
]

subprocess.run(cmd, check=True)


subprocess.run("rm -rf aws awscliv2.zip", shell=True, check=True)
