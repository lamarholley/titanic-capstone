"""
Download the Titanic dataset used by this project.

Data files are intentionally excluded from version control (see
.gitignore / README), so this script re-fetches the raw CSV on setup.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

# Public mirror of the classic Kaggle "Titanic - Machine Learning from
# Disaster" training set (891 labeled passengers). See README.md for
# why this mirror is used instead of the Kaggle API (which requires an
# authenticated account and was not available in the build environment).
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DEST = Path("data/raw/titanic.csv")


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Titanic dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, DEST)
    size_kb = DEST.stat().st_size / 1024
    print(f"Saved to {DEST} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
