#!/usr/bin/env python3
"""
Replace NaN / Inf in SPANet HDF5 INPUTS/Jet/btag with a finite fill value (default 0.0).

Uses chunked read–modify–write so large merged files stay memory-safe.

Examples (review before running):

  # Copy train to a new file, fix the copy (original unchanged)
  python scripts/fix_h5_jet_btag_nan_to_zero.py train.h5 -o train_btag_fixed.h5

  # Fix several files into a directory (full copy per file, then patch)
  python scripts/fix_h5_jet_btag_nan_to_zero.py a.h5 b.h5 --output-dir /path/to/out/

  # Overwrite files in place (no extra disk for a second full copy)
  python scripts/fix_h5_jet_btag_nan_to_zero.py train.h5 --in-place

  # Only report how many values would change
  python scripts/fix_h5_jet_btag_nan_to_zero.py train.h5 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_DATASET = "INPUTS/Jet/btag"


def count_and_replace_nonfinite(
    path: Path,
    *,
    dataset_path: str,
    fill_value: float,
    chunk_rows: int,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Returns (n_bad_values_replaced, n_elements_scanned).
    """
    n_bad = 0
    n_total = 0

    with h5py.File(path, "r+" if not dry_run else "r") as f:
        if dataset_path not in f:
            raise KeyError(f"Dataset not found: {dataset_path} in {path}")
        ds = f[dataset_path]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"Not a dataset: {dataset_path}")
        if not np.issubdtype(ds.dtype, np.floating):
            raise TypeError(f"Expected floating dtype, got {ds.dtype}")

        shape = ds.shape
        if len(shape) == 0:
            n_total = 1
            val = np.array(ds[()])
            if not np.isfinite(val):
                n_bad = 1
                if not dry_run:
                    ds[()] = np.asarray(fill_value, dtype=ds.dtype)
            return n_bad, n_total

        n0 = shape[0]
        for row in range(0, n0, chunk_rows):
            end = min(row + chunk_rows, n0)
            sl = (slice(row, end),) + (slice(None),) * (len(shape) - 1)
            chunk = np.asarray(ds[sl], dtype=np.float64)
            n_total += chunk.size
            bad = ~np.isfinite(chunk)
            n_here = int(bad.sum())
            if n_here == 0:
                continue
            n_bad += n_here
            if not dry_run:
                chunk = chunk.copy()
                chunk[bad] = fill_value
                ds[sl] = chunk.astype(ds.dtype, copy=False)

    return n_bad, n_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set non-finite INPUTS/Jet/btag values to 0.0 (or --fill-value)."
    )
    parser.add_argument(
        "h5_files",
        nargs="+",
        type=Path,
        help="One or more SPANet-style HDF5 files.",
    )
    out_group = parser.add_mutually_exclusive_group(required=False)
    out_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write a single fixed file (use with exactly one input file).",
    )
    out_group.add_argument(
        "--output-dir",
        type=Path,
        help="Copy each input here (same basename), then patch the copy.",
    )
    out_group.add_argument(
        "--in-place",
        action="store_true",
        help="Modify each input file without making a full-file copy first.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HDF5 path to the btag array (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help="Value used for NaN and Inf (default: 0.0).",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="Leading-dimension chunk size (default: 500000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count non-finite values; do not write.",
    )
    args = parser.parse_args()

    if args.output is not None and len(args.h5_files) != 1:
        print("error: -o/--output requires exactly one input file", file=sys.stderr)
        sys.exit(2)

    mode_count = (
        (args.output is not None)
        + (args.output_dir is not None)
        + (1 if args.in_place else 0)
    )
    if not args.dry_run and mode_count != 1:
        print(
            "error: specify exactly one of -o, --output-dir, or --in-place "
            "(or use --dry-run alone to only count on the source file)",
            file=sys.stderr,
        )
        sys.exit(2)

    exit_code = 0
    for src in args.h5_files:
        if not src.is_file():
            print(f"error: not a file: {src}", file=sys.stderr)
            exit_code = 1
            continue

        # File to open: dry-run always reads the source; otherwise copy first or edit in place.
        work_path = src
        describe: str
        if args.dry_run and not args.in_place and args.output is None and args.output_dir is None:
            work_path = src
            describe = f"dry-run (read {src})"
        elif args.dry_run and args.in_place:
            work_path = src
            describe = f"dry-run (read {src}, would edit in place)"
        elif args.dry_run and args.output is not None:
            work_path = src
            describe = f"dry-run (read {src}; would write to {args.output})"
        elif args.dry_run and args.output_dir is not None:
            work_path = src
            describe = f"dry-run (read {src}; would copy to {args.output_dir / src.name})"
        elif args.output is not None:
            work_path = args.output
            shutil.copy2(src, work_path)
            describe = str(work_path)
        elif args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            work_path = args.output_dir / src.name
            shutil.copy2(src, work_path)
            describe = str(work_path)
        else:
            # --in-place
            work_path = src
            describe = str(work_path)

        try:
            n_bad, n_tot = count_and_replace_nonfinite(
                work_path,
                dataset_path=args.dataset,
                fill_value=args.fill_value,
                chunk_rows=args.chunk_rows,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"error processing {src} ({describe}): {e}", file=sys.stderr)
            exit_code = 1
            continue

        pct = (100.0 * n_bad / n_tot) if n_tot else 0.0
        print(f"{src}")
        print(f"  {describe}")
        print(f"  non-finite values: {n_bad} / {n_tot} ({pct:.6g}%)")
        if args.dry_run:
            print("  (no changes written)")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
