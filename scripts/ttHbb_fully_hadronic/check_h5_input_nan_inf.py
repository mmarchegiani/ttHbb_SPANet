#!/usr/bin/env python3
"""
Scan SPANet-style HDF5 files for NaN / Inf in input feature arrays (and optionally weights).

Typical layout: INPUTS/<Source>/<feature> with float datasets, plus MASK where present.

Examples:
  python scripts/check_h5_input_nan_inf.py \\
    /path/to/merged_train.h5 /path/to/merged_test.h5

  python scripts/check_h5_input_nan_inf.py --from-options \\
    options_files/ttHbb_fully_hadronic/classifier/options_file_Run2_Run3_sig_QCD_classifier_btag_full.json

  # Quick smoke test (first 200k events only):
  python scripts/check_h5_input_nan_inf.py --max-events 200000 train.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


@dataclass
class DatasetReport:
    path: str
    shape: tuple[int, ...]
    dtype: str
    nan_count: int = 0
    pos_inf_count: int = 0
    neg_inf_count: int = 0
    finite_count: int = 0
    example_indices: list[tuple[int, ...]] = field(default_factory=list)

    @property
    def bad_count(self) -> int:
        return self.nan_count + self.pos_inf_count + self.neg_inf_count

    def add_chunk_stats(
        self,
        chunk: np.ndarray,
        global_offset: tuple[int, ...],
        collect_examples: int,
    ) -> None:
        """Update counts from a chunk; global_offset is the index of chunk[0,...] in the full dataset."""
        flat = chunk.reshape(-1)
        n = flat.size
        if n == 0:
            return
        is_nan = np.isnan(flat)
        is_pos_inf = np.isposinf(flat)
        is_neg_inf = np.isneginf(flat)
        self.nan_count += int(is_nan.sum())
        self.pos_inf_count += int(is_pos_inf.sum())
        self.neg_inf_count += int(is_neg_inf.sum())
        self.finite_count += int(np.isfinite(flat).sum())

        if collect_examples > 0 and len(self.example_indices) < collect_examples:
            bad = is_nan | is_pos_inf | is_neg_inf
            bad_flat = np.flatnonzero(bad)
            for i in bad_flat:
                if len(self.example_indices) >= collect_examples:
                    break
                idx = np.unravel_index(int(i), chunk.shape)
                full_idx = tuple(int(global_offset[d] + idx[d]) for d in range(chunk.ndim))
                self.example_indices.append(full_idx)


def iter_float_datasets(group: h5py.Group, prefix: str = "") -> Iterator[tuple[str, h5py.Dataset]]:
    for key in group.keys():
        obj = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(obj, h5py.Dataset):
            if np.issubdtype(obj.dtype, np.floating):
                yield path, obj
        elif isinstance(obj, h5py.Group):
            yield from iter_float_datasets(obj, path)


def scan_dataset(
    dataset: h5py.Dataset,
    *,
    chunk_rows: int,
    max_events: int | None,
    collect_examples: int,
) -> DatasetReport:
    shape = dataset.shape
    dtype = str(dataset.dtype)
    report = DatasetReport(path=dataset.name, shape=shape, dtype=dtype)

    if len(shape) == 0:
        chunk = np.zeros((), dtype=dataset.dtype)
        dataset.read_direct(chunk)
        arr = np.array([chunk])
        report.add_chunk_stats(arr, (0,), collect_examples)
        return report

    # Chunk along leading dimension (events)
    n0 = shape[0]
    limit = n0 if max_events is None else min(n0, max_events)

    row = 0
    while row < limit:
        end = min(row + chunk_rows, limit)
        sl = (slice(row, end),) + (slice(None),) * (len(shape) - 1)
        chunk = np.asarray(dataset[sl], dtype=np.float64)
        lead_offset = (row,) + (0,) * (max(len(shape) - 1, 0))
        report.add_chunk_stats(chunk, lead_offset, collect_examples)
        row = end

    return report


def scan_file(
    h5_path: Path,
    *,
    root_group: str,
    also_weights: bool,
    chunk_rows: int,
    max_events: int | None,
    collect_examples: int,
) -> list[DatasetReport]:
    reports: list[DatasetReport] = []
    with h5py.File(h5_path, "r") as f:
        groups_to_scan: list[h5py.Group] = []
        if root_group in f:
            if isinstance(f[root_group], h5py.Group):
                groups_to_scan.append(f[root_group])
        if also_weights and "WEIGHTS" in f and isinstance(f["WEIGHTS"], h5py.Group):
            groups_to_scan.append(f["WEIGHTS"])

        for g in groups_to_scan:
            for _rel_path, ds in iter_float_datasets(g):
                reports.append(
                    scan_dataset(
                        ds,
                        chunk_rows=chunk_rows,
                        max_events=max_events,
                        collect_examples=collect_examples,
                    )
                )
    return reports


def print_report(h5_path: Path, reports: list[DatasetReport]) -> int:
    """Return number of datasets with any non-finite values."""
    bad_datasets = 0
    print(f"\n{'=' * 72}")
    print(f"File: {h5_path}")
    print(f"{'=' * 72}")
    for r in sorted(reports, key=lambda x: x.path):
        if r.bad_count == 0:
            continue
        bad_datasets += 1
        total = r.bad_count + r.finite_count
        frac = (r.bad_count / total * 100) if total else 0.0
        print(f"\n  {r.path}")
        print(f"    shape={r.shape} dtype={r.dtype}")
        print(
            f"    non-finite: {r.bad_count} ({frac:.6g}%)  "
            f"[nan={r.nan_count}, +inf={r.pos_inf_count}, -inf={r.neg_inf_count}]"
        )
        if r.example_indices:
            print(f"    example flat/multi-indices (first {len(r.example_indices)}):")
            for idx in r.example_indices[:20]:
                print(f"      {idx}")
            if len(r.example_indices) > 20:
                print(f"      ... ({len(r.example_indices) - 20} more)")

    ok = [r for r in reports if r.bad_count == 0]
    if bad_datasets == 0:
        print(f"\n  No NaN or Inf in any of the {len(reports)} scanned float dataset(s).")
    elif ok:
        print(
            f"\n  Clean ({len(ok)} dataset(s)): {', '.join(r.path for r in ok)}"
        )
    return bad_datasets


def load_paths_from_options_json(path: Path) -> list[Path]:
    with path.open() as f:
        opts: dict[str, Any] = json.load(f)
    out: list[Path] = []
    for key in ("training_file", "validation_file", "testing_file"):
        p = opts.get(key)
        if p and isinstance(p, str) and p.strip():
            out.append(Path(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan HDF5 input feature arrays for NaN and Inf values."
    )
    parser.add_argument(
        "h5_files",
        nargs="*",
        type=Path,
        help="HDF5 file(s) to scan (SPANet merged dataset format).",
    )
    parser.add_argument(
        "--from-options",
        type=Path,
        default=None,
        help="JSON options file with training_file / validation_file paths (same as spanet.train -of).",
    )
    parser.add_argument(
        "--root",
        default="INPUTS",
        help="HDF5 group to scan recursively for float datasets (default: INPUTS).",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Also scan WEIGHTS/ float datasets.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="Rows to read per chunk along the leading axis (default: 500000).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Only scan the first N events (leading dimension) for a quick check.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=16,
        help="Max number of example (multi-)indices to print per bad dataset (default: 16).",
    )
    args = parser.parse_args()

    paths: list[Path] = list(args.h5_files)
    if args.from_options:
        paths.extend(load_paths_from_options_json(args.from_options))

    # De-dupe while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    paths = uniq

    if not paths:
        print("No HDF5 files given. Pass file paths or --from-options <options.json>.", file=sys.stderr)
        sys.exit(2)

    total_bad = 0
    for h5_path in paths:
        if not h5_path.is_file():
            print(f"WARNING: missing file, skipping: {h5_path}", file=sys.stderr)
            continue
        reports = scan_file(
            h5_path,
            root_group=args.root,
            also_weights=args.include_weights,
            chunk_rows=args.chunk_rows,
            max_events=args.max_events,
            collect_examples=args.examples,
        )
        total_bad += print_report(h5_path, reports)

    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
