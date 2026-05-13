#!/usr/bin/env python3
"""
Jet multiplicity sanity checks for parquet and/or H5.

- Parquet: counts jets from a jagged collection (default: JetGood).
- H5: counts jets from INPUTS/Jet/MASK (sum over axis=1).

Outputs:
- Printed stats (min/max/mean and a few quantiles)
- Histogram PNG (if matplotlib available)
- Histogram counts CSV (always)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def summarize(name: str, counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=np.int64)
    q = np.quantile(counts, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    payload = {
        "name": name,
        "n_events": int(counts.shape[0]),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "p50": float(q[1]),
        "p90": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "frac_gt16": float(np.mean(counts > 16)),
        "frac_ge16": float(np.mean(counts >= 16)),
    }
    return payload


def counts_from_h5(h5_path: str, mask_key: str = "INPUTS/Jet/MASK") -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as f:
        if mask_key not in f:
            raise KeyError(f"{h5_path}: missing {mask_key}")
        mask = np.asarray(f[mask_key][:]).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"{h5_path}: expected mask ndim=2, got {mask.ndim} with shape {mask.shape}")
    return mask.sum(axis=1).astype(np.int64)


def counts_from_parquet(parquet_paths: list[str], collection: str = "JetGood") -> np.ndarray:
    import awkward as ak

    arrays = []
    for p in parquet_paths:
        arr = ak.from_parquet(p)
        if collection not in arr.fields:
            raise KeyError(f"{p}: missing top-level field '{collection}'. Available: {arr.fields}")
        arrays.append(ak.num(arr[collection], axis=1))
    if len(arrays) == 1:
        out = arrays[0]
    else:
        out = ak.concatenate(arrays, axis=0)
    return np.asarray(out, dtype=np.int64)


def save_hist_csv(out_csv: Path, counts: np.ndarray, bins: np.ndarray) -> None:
    hist, edges = np.histogram(counts, bins=bins)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("bin_left,bin_right,count\n")
        for i in range(len(hist)):
            f.write(f"{edges[i]},{edges[i+1]},{int(hist[i])}\n")


def maybe_save_png(out_png: Path, series: list[tuple[str, np.ndarray]], bins: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib not available; skipping PNG. ({exc})")
        return

    plt.figure(figsize=(8, 5.5))
    for name, counts in series:
        plt.hist(counts, bins=bins, histtype="step", linewidth=2, label=f"{name} (N={len(counts):,})")
    plt.xlabel("Number of jets per event")
    plt.ylabel("Events")
    plt.yscale("log")
    plt.title("Jet multiplicity")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="*", default=[], help="Input parquet file(s)")
    ap.add_argument("--parquet_collection", default="JetGood", help="Jet collection name in parquet (default: JetGood)")
    ap.add_argument("--h5", nargs="*", default=[], help="Input H5 file(s)")
    ap.add_argument("--h5_mask_key", default="INPUTS/Jet/MASK", help="Mask dataset key in H5")
    ap.add_argument("--out_dir", required=True, help="Output directory for reports")
    ap.add_argument("--max_bin", type=int, default=40, help="Max jet-multiplicity bin edge (default: 40)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = np.arange(0, int(args.max_bin) + 2) - 0.5  # integer-centered bins

    series: list[tuple[str, np.ndarray]] = []

    if args.parquet:
        counts_pq = counts_from_parquet(args.parquet, collection=args.parquet_collection)
        series.append((f"parquet:{args.parquet_collection}", counts_pq))
        stats = summarize(f"parquet:{args.parquet_collection}", counts_pq)
        print(stats)
        save_hist_csv(out_dir / "jet_mult_parquet.csv", counts_pq, bins)

    if args.h5:
        # If multiple h5 files, concatenate counts.
        all_counts = []
        for p in args.h5:
            all_counts.append(counts_from_h5(p, mask_key=args.h5_mask_key))
        counts_h5 = np.concatenate(all_counts, axis=0) if len(all_counts) > 1 else all_counts[0]
        series.append(("h5:MASK", counts_h5))
        stats = summarize("h5:MASK", counts_h5)
        print(stats)
        save_hist_csv(out_dir / "jet_mult_h5.csv", counts_h5, bins)

    if not series:
        raise SystemExit("Provide at least one of --parquet or --h5")

    maybe_save_png(out_dir / "jet_multiplicity.png", series, bins)

    # Quick “loss if cap at 16” report
    for name, counts in series:
        over = int(np.sum(counts > 16))
        print(f"{name}: events with >16 jets = {over:,} / {len(counts):,} ({over/len(counts):.4%})")


if __name__ == "__main__":
    main()

