from pathlib import Path
import sys
import pandas as pd
import csv
from pprint import pprint

#!/usr/bin/env python3
"""
read_csv.py

Read data/echodyna/FileList.csv and return a pandas DataFrame.
"""


def read_filelist(root: str = "data/echodyna", name: str = "FileList.csv"):
    p = Path(root) / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        # fallback to built-in csv module -> returns list of dicts
        with p.open(newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            return list(reader)

if __name__ == "__main__":
    try:
        data = read_filelist()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # If pandas DataFrame, print summary; otherwise print first 5 records
    if hasattr(data, "head"):
        print(f"Loaded DataFrame with {len(data)} rows and {len(data.columns)} columns")
        print(data.head().to_string(index=False))
    else:
        print(f"Loaded {len(data)} records")
        pprint(data[:5])

    # Get the "NumberOfFrames" column and print the min, mean, median, max
    if hasattr(data, "get"):
        try:
            num_frames = pd.to_numeric(data.get("NumberOfFrames"), errors='coerce').dropna()
            print("NumberOfFrames statistics:")
            print(f"  Min: {num_frames.min()}")
            print(f"  Mean: {num_frames.mean()}")
            print(f"  Median: {num_frames.median()}")
            print(f"  Max: {num_frames.max()}")
        except Exception as e:
            print(f"Error processing NumberOfFrames column: {e}", file=sys.stderr)