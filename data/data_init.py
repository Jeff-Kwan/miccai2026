#!/usr/bin/env python3
"""
Download AzCopy v10 (official aka.ms link) and use it to download a blob container
given a SAS URL.

Example:
  python download_echonet.py --dest ./echonetdynamic-2
"""

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Official AzCopy v10 download entrypoints (they redirect to the latest release)
# Microsoft docs describe using these aka.ms download links. :contentReference[oaicite:0]{index=0}
AZCOPY_URLS = {
    "Windows": "https://aka.ms/downloadazcopy-v10-windows",  # zip
    "Linux": "https://aka.ms/downloadazcopy-v10-linux",      # tar.gz
    "Darwin": "https://aka.ms/downloadazcopy-v10-mac",       # zip or tar.gz depending on current packaging
}

SOURCE_SAS_URL = (
    "https://aimistanforddatasets01.blob.core.windows.net/echonetdynamic-2"
    "?sv=2019-02-02&sr=c&sig=Z5d6ddT9LpjH7Hdr72QFUBravb9VwYOvFAiOAkTEsQA%3D"
    "&st=2026-01-27T09%3A55%3A30Z&se=2026-02-26T10%3A00%3A30Z&sp=rl"
)

def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)

def extract_archive(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)

    # Try zip first
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
        return

    # Try tar variants
    try:
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(extract_to)
        return
    except tarfile.TarError:
        pass

    raise RuntimeError(f"Unknown archive format: {archive_path}")

def find_azcopy_executable(root: Path) -> Path:
    candidates = []
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() in ("azcopy", "azcopy.exe"):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Could not find azcopy executable under: {root}")

    # Prefer top-level-ish azcopy
    candidates.sort(key=lambda p: (len(p.parts), p.name.lower() != "azcopy.exe"))
    return candidates[0]

def ensure_executable(path: Path) -> None:
    if os.name != "nt":
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def run_azcopy(azcopy_path: Path, source_url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(azcopy_path),
        "copy",
        source_url,
        str(dest_dir),
        "--recursive=true",
    ]
    print("Running:", " ".join([cmd[0], cmd[1], "<SAS_URL>", cmd[3], cmd[4]]))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="./data/echodyna", help="Destination folder")
    ap.add_argument("--source", default=SOURCE_SAS_URL, help="Source SAS URL")
    ap.add_argument("--azcopy-dir", default=None, help="Optional cache directory for AzCopy")
    args = ap.parse_args()

    sys_platform = platform.system()
    if sys_platform not in AZCOPY_URLS:
        raise RuntimeError(f"Unsupported OS: {sys_platform}")

    cache_dir = Path(args.azcopy_dir) if args.azcopy_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)

        # If caching is enabled and azcopy already exists, reuse it
        if cache_dir and (cache_dir / ("azcopy.exe" if os.name == "nt" else "azcopy")).exists():
            azcopy = cache_dir / ("azcopy.exe" if os.name == "nt" else "azcopy")
        else:
            url = AZCOPY_URLS[sys_platform]
            archive = work / "azcopy_download"
            print(f"Downloading AzCopy from: {url}")
            download_file(url, archive)

            extracted = work / "azcopy_extracted"
            extract_archive(archive, extracted)

            azcopy = find_azcopy_executable(extracted)
            ensure_executable(azcopy)

            if cache_dir:
                target = cache_dir / ("azcopy.exe" if azcopy.name.lower().endswith(".exe") else "azcopy")
                shutil.copy2(azcopy, target)
                ensure_executable(target)
                azcopy = target

        print(f"Using AzCopy at: {azcopy}")
        run_azcopy(azcopy, args.source, Path(args.dest))
        print("Done.")

if __name__ == "__main__":
    main()