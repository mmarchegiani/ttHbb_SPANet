#!/usr/bin/env python3
import argparse
import h5py
import numpy as np


def sanitize_dataset(ds, ge_value=16, replacement=-1, chunk_size=2_000_000):
    if not isinstance(ds.shape, tuple) or len(ds.shape) < 1:
        raise ValueError(f"Unexpected dataset shape: {ds.shape} for {ds.name}")
    n = ds.shape[0]
    # Ensure we have an integer dataset to work with
    if np.issubdtype(ds.dtype, np.floating):
        raise TypeError(f"Expected integer dtype for {ds.name}, got {ds.dtype}")

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        arr = ds[start:end]
        # In-place sanitize chunk
        mask = arr >= ge_value
        if np.any(mask):
            arr[mask] = replacement
            ds[start:end] = arr


def main():
    parser = argparse.ArgumentParser(
        description="Set any SPANet target indices >= 16 to -1 in H5 files."
    )
    parser.add_argument("h5_files", nargs="+", help="H5 files to patch in place")
    parser.add_argument("--ge", type=int, default=16, help="Threshold (>= ge -> replacement)")
    parser.add_argument("--replacement", type=int, default=-1, help="Replacement value")
    parser.add_argument("--chunk_size", type=int, default=2_000_000, help="Elements per chunk along axis 0")
    args = parser.parse_args()

    for path in args.h5_files:
        with h5py.File(path, "a") as f:
            if "TARGETS" not in f:
                print(f"SKIP (no TARGETS group): {path}")
                continue

            targets_group = f["TARGETS"]
            parents = list(targets_group.keys())
            patched_any = False

            for parent in parents:
                parent_group = targets_group[parent]
                for ds_name in parent_group.keys():
                    ds = parent_group[ds_name]
                    # Fast check: sample min/max over first chunk might miss outliers,
                    # so compute full max on the dataset name is too expensive; rely on chunk scan.
                    sanitize_dataset(ds, ge_value=args.ge, replacement=args.replacement, chunk_size=args.chunk_size)
                    patched_any = True

            print(f"Patched TARGETS in: {path} (ge>={args.ge} -> {args.replacement})")


if __name__ == "__main__":
    main()

