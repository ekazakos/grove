#!/usr/bin/env python3
"""
Split a pickled dictionary into individual pickle files
(one file per key) inside a target directory.

Usage
-----
python split_dict_pkl.py data.pkl out_dir
"""

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump each annotation of a pickled dict containing HowToGround1M/iGround annotations into its own .pkl file."
    )
    parser.add_argument(
        "input_pkl",
        type=Path,
        help="Path to the source .pkl file.",
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory to write individual .pkl files to.",
    )
    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)

    with args.input_pkl.open("rb") as f:
        data = pickle.load(f)

    for key, value in data.items():
        filename = str(key) + ".pkl"
        out_path = args.target_dir / filename
        with out_path.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {len(data)} pickle files to {args.target_dir.resolve()}")


if __name__ == "__main__":
    main()