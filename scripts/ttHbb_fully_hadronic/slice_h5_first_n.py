#!/usr/bin/env python3
"""
Create a smaller SPANet-style merged HDF5 by keeping only the first N events.

This copies the entire HDF5 tree (groups, datasets, attrs), but for datasets whose
leading dimension equals the event count, it slices axis-0 to [:N] using chunked
read/write (memory-safe).

Typical SPANet merged files have datasets like:
  INPUTS/Jet/pt     (N, 16)
  TARGETS/h/b1      (N,)
  WEIGHTS/weight    (N,)
and scalar metadata datasets (no leading N) which are copied as-is.

Example:
  python scripts/dataset/slice_h5_first_n.py \
    -i /path/to/merged_test_*.h5 \
    -o /path/to/merged_test_first1M.h5 \
    -n 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


def infer_num_events(h5: h5py.File) -> int:
    # Prefer WEIGHTS/weight as a reliable event-count dataset.
    if "WEIGHTS" in h5 and "weight" in h5["WEIGHTS"]:
        ds = h5["WEIGHTS"]["weight"]
        if isinstance(ds, h5py.Dataset) and ds.ndim >= 1:
            return int(ds.shape[0])

    # Fall back: find the maximum leading dimension among datasets.
    n = 0
    def visit(_name, obj):
        nonlocal n
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            n = max(n, int(obj.shape[0]))
    h5.visititems(visit)
    if n <= 0:
        raise RuntimeError("Could not infer number of events (no 1D+ datasets found).")
    return n


def copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def dataset_create_kwargs(
    src: h5py.Dataset,
    *,
    compression: Optional[str],
    compression_opts: Optional[int],
) -> dict:
    kwargs: dict = {}
    # Preserve chunking if present; else let h5py decide.
    if src.chunks is not None:
        kwargs["chunks"] = src.chunks
    if compression and compression.lower() != "none":
        kwargs["compression"] = compression
        if compression_opts is not None and compression.lower() == "gzip":
            kwargs["compression_opts"] = int(compression_opts)
    return kwargs


def copy_dataset_sliced(
    src: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    *,
    n_events_total: int,
    n_keep: int,
    chunk_rows: int,
    compression: Optional[str],
    compression_opts: Optional[int],
) -> None:
    # Decide whether this dataset is event-shaped.
    event_shaped = src.ndim >= 1 and int(src.shape[0]) == int(n_events_total)

    if event_shaped:
        out_shape = (n_keep,) + src.shape[1:]
        out_ds = dst_group.create_dataset(
            name,
            shape=out_shape,
            dtype=src.dtype,
            **dataset_create_kwargs(src, compression=compression, compression_opts=compression_opts),
        )
        copy_attrs(src, out_ds)

        # Chunked copy along axis-0
        for start in range(0, n_keep, chunk_rows):
            end = min(start + chunk_rows, n_keep)
            sl = (slice(start, end),) + (slice(None),) * (src.ndim - 1)
            out_ds[sl] = src[sl]
    else:
        # Copy full dataset as-is (small metadata, or non-event-shaped arrays).
        data = src[()]
        out_ds = dst_group.create_dataset(
            name,
            data=data,
            dtype=src.dtype,
            **dataset_create_kwargs(src, compression=compression, compression_opts=compression_opts),
        )
        copy_attrs(src, out_ds)


def copy_group_recursive(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    *,
    n_events_total: int,
    n_keep: int,
    chunk_rows: int,
    compression: Optional[str],
    compression_opts: Optional[int],
) -> None:
    copy_attrs(src_group, dst_group)
    for key, item in src_group.items():
        if isinstance(item, h5py.Group):
            g = dst_group.create_group(key)
            copy_group_recursive(
                item,
                g,
                n_events_total=n_events_total,
                n_keep=n_keep,
                chunk_rows=chunk_rows,
                compression=compression,
                compression_opts=compression_opts,
            )
        elif isinstance(item, h5py.Dataset):
            copy_dataset_sliced(
                item,
                dst_group,
                key,
                n_events_total=n_events_total,
                n_keep=n_keep,
                chunk_rows=chunk_rows,
                compression=compression,
                compression_opts=compression_opts,
            )
        else:
            raise TypeError(f"Unsupported HDF5 item type at {item.name}: {type(item)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice a merged SPANet HDF5 to the first N events.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input .h5 file")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output .h5 file to create")
    parser.add_argument("-n", "--max-events", required=True, type=int, help="Keep only the first N events")
    parser.add_argument("--chunk-rows", type=int, default=200_000, help="Chunk size along axis-0 (default: 200000)")
    parser.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=["none", "gzip", "lzf"],
        help="Optional compression for output datasets (default: none)",
    )
    parser.add_argument(
        "--compression-opts",
        type=int,
        default=4,
        help="gzip level (1-9) when --compression gzip (default: 4)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    args = parser.parse_args()

    if args.max_events <= 0:
        raise SystemExit("error: --max-events must be > 0")

    if args.output.exists():
        if args.overwrite:
            args.output.unlink()
        else:
            raise SystemExit(f"error: output exists (use --overwrite): {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as src:
        n_total = infer_num_events(src)
        n_keep = min(int(args.max_events), n_total)
        with h5py.File(args.output, "w") as dst:
            dst.attrs["sliced_from"] = str(Path(args.input).resolve())
            dst.attrs["sliced_max_events"] = int(args.max_events)
            dst.attrs["sliced_kept_events"] = int(n_keep)
            dst.attrs["sliced_total_events"] = int(n_total)

            copy_group_recursive(
                src,
                dst,
                n_events_total=n_total,
                n_keep=n_keep,
                chunk_rows=int(args.chunk_rows),
                compression=None if args.compression == "none" else args.compression,
                compression_opts=int(args.compression_opts),
            )

    print(f"Wrote: {args.output} (kept {n_keep:,} / {n_total:,} events)")


if __name__ == "__main__":
    main()

