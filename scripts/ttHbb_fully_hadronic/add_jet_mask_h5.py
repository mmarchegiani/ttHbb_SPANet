#!/usr/bin/env python3
"""Add INPUTS/Jet/MASK to SPANet H5 if missing (MASK = ~(pt == 0)), matching parquet_to_h5 behavior."""
import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_files", nargs="+", help="One or more .h5 files to patch in place")
    args = parser.parse_args()

    for path in args.h5_files:
        with h5py.File(path, "a") as f:
            g = f["INPUTS/Jet"]
            if "MASK" in g:
                print(f"SKIP (already has MASK): {path}")
                continue
            if "pt" not in g:
                raise KeyError(f"{path}: INPUTS/Jet/pt missing; cannot build MASK")
            pt = np.asarray(g["pt"][:])
            mask = ~(pt == 0)
            g.create_dataset("MASK", data=mask.astype(np.bool_))
            print(f"OK: wrote INPUTS/Jet/MASK shape={mask.shape}: {path}")


if __name__ == "__main__":
    main()
