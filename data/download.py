"""
data/download.py
-----------------
Instructions and utilities for downloading datasets used in the paper.

No raw data is bundled in this repository to keep it lightweight and
avoid licensing issues.  Run this script to download datasets.

Datasets:
  1. NSL-KDD    — network intrusion detection benchmark
  2. CICIDS2017 — Canadian Institute for Cybersecurity dataset (optional)

Usage:
    python data/download.py --dataset nsl_kdd
    python data/download.py --all
"""

import argparse
import os
import urllib.request
import zipfile


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    "nsl_kdd": {
        "description": "NSL-KDD network intrusion detection dataset",
        "url": "https://www.unb.ca/cic/datasets/nsl.html",
        "files": {
            "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
            "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
        },
        "note": (
            "NSL-KDD is freely available.  If the direct links above are down, "
            "download manually from https://www.unb.ca/cic/datasets/nsl.html "
            "and place in data/."
        ),
    },
}

# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

def download_file(url: str, dest: str, timeout: int = 30) -> bool:
    """Download a single file with progress indication.

    Parameters
    ----------
    url : str
    dest : str
        Destination file path.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if os.path.exists(dest):
        print(f"  [SKIP] Already exists: {os.path.basename(dest)}")
        return True

    print(f"  Downloading {os.path.basename(dest)} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"done  ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def download_dataset(name: str) -> None:
    """Download a named dataset.

    Parameters
    ----------
    name : str
        One of the keys in DATASETS.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")

    info = DATASETS[name]
    print(f"\n{'=' * 60}")
    print(f"  {info['description']}")
    print(f"{'=' * 60}")

    all_ok = True
    for filename, url in info["files"].items():
        dest = os.path.join(DATA_DIR, filename)
        ok = download_file(url, dest)
        all_ok = all_ok and ok

    if all_ok:
        print(f"\n  All files downloaded to: {DATA_DIR}/")
    else:
        print(f"\n  Some downloads failed.  {info['note']}")


def list_datasets() -> None:
    """Print available datasets."""
    print("\nAvailable datasets:")
    for name, info in DATASETS.items():
        print(f"  {name:15s}  {info['description']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for HED experiments.")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS),
                        help="Name of dataset to download")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.all:
        for name in DATASETS:
            download_dataset(name)
    elif args.dataset:
        download_dataset(args.dataset)
    else:
        parser.print_help()
