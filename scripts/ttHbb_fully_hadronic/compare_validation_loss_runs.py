#!/usr/bin/env python3
"""
Overlay validation loss curves from multiple training runs (CMS-style).

Inputs are the extracted-metrics CSVs produced under:
  /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/<run>/.../version_X/

For each run, we use:
  - epoch.csv to map step -> epoch
  - validation_loss__total_loss.csv (by default) for the metric values

Example:
  python scripts/ttHbb_fully_hadronic/compare_validation_loss_runs.py \
    --runs \
      /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_T/classifier_btag_T/version_1 \
      /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_TM/classifier_btag_TM/version_4 \
    --labels btag_T btag_TM \
    --title "Validation total loss" \
    --cms-subtitle "Internal work" \
    --max-epoch 25 \
    --out /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/common_plots/compare_val_total_loss.png

You can also pass explicit metric CSV paths with --metric-csvs, and override
which validation loss to plot with --metric-filename (e.g. validation_loss__h__assignment_loss.csv).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _import_plotting():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import mplhep as hep  # type: ignore

        return plt, hep
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing plotting dependencies. Install in your env with:\n"
            "  python -m pip install matplotlib mplhep\n"
            f"Original error: {e}"
        ) from e


@dataclass(frozen=True)
class Series:
    epochs: np.ndarray
    values: np.ndarray


EPOCH_FILENAME = "epoch.csv"


def read_metric_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    steps: List[int] = []
    vals: List[float] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            vals.append(float(row["value"]))
    return np.asarray(steps, dtype=np.int64), np.asarray(vals, dtype=np.float64)


def steps_to_epochs(metric_steps: np.ndarray, epoch_steps: np.ndarray, epoch_values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(epoch_steps, metric_steps, side="right") - 1
    idx = np.clip(idx, 0, len(epoch_steps) - 1)
    return epoch_values[idx].astype(np.int64)


def aggregate_by_epoch(epochs: np.ndarray, values: np.ndarray) -> Series:
    order = np.argsort(epochs)
    epochs = epochs[order]
    values = values[order]

    uniq = np.unique(epochs)
    out = np.empty_like(uniq, dtype=np.float64)
    for i, e in enumerate(uniq):
        m = epochs == e
        out[i] = float(values[m].mean())
    return Series(epochs=uniq, values=out)


def cap_series(series: Series, *, max_epoch: int | None) -> Series:
    if max_epoch is None:
        return series
    m = series.epochs <= max_epoch
    return Series(epochs=series.epochs[m], values=series.values[m])


def derive_default_label(run_dir: Path) -> str:
    parts = list(run_dir.parts)
    if "training_metrics" in parts:
        i = parts.index("training_metrics")
        rel = parts[i + 1 :]
        if rel:
            if len(rel) >= 2 and rel[-1].startswith("version_"):
                return f"{rel[-2]}/{rel[-1]}"
            return "/".join(rel[-3:])
    return run_dir.name


def add_internal_label(ax, *, subtitle: str) -> None:
    ax.text(
        0.02,
        0.98,
        "CMS",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.90,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )


def auto_ylim_for_losses(series_list: List[Series]) -> tuple[float, float]:
    ys: List[np.ndarray] = [s.values for s in series_list if s.values.size]
    if not ys:
        return 0.0, 1.0
    y = np.concatenate(ys)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.0

    lo = float(np.percentile(y, 1))
    hi = float(np.percentile(y, 99))
    span = max(hi - lo, 1e-9)
    pad = 0.05 * span
    lo2 = lo - pad
    hi2 = hi + pad
    if hi2 <= lo2:
        lo2 = float(np.min(y))
        hi2 = float(np.max(y))
        span2 = max(hi2 - lo2, 1e-9)
        lo2 -= 0.05 * span2
        hi2 += 0.05 * span2
    return lo2, hi2


def load_series_from_run(run_dir: Path, *, metric_filename: str, max_epoch: int | None) -> Series:
    epoch_csv = run_dir / EPOCH_FILENAME
    metric_csv = run_dir / metric_filename
    if not epoch_csv.is_file():
        raise FileNotFoundError(f"Missing {EPOCH_FILENAME} in {run_dir}")
    if not metric_csv.is_file():
        raise FileNotFoundError(f"Missing {metric_filename} in {run_dir}")

    epoch_steps, epoch_vals = read_metric_csv(epoch_csv)
    metric_steps, metric_vals = read_metric_csv(metric_csv)

    epochs = steps_to_epochs(metric_steps, epoch_steps, epoch_vals)
    series = aggregate_by_epoch(epochs, metric_vals)
    return cap_series(series, max_epoch=max_epoch)


def load_series_from_metric_csv(metric_csv: Path, *, max_epoch: int | None) -> tuple[Path, str]:
    run_dir = metric_csv.parent
    return run_dir, metric_csv.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay validation loss curves from multiple runs (CMS style).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", nargs="+", type=Path, help="Run directories containing epoch.csv and metric CSV.")
    group.add_argument("--metric-csvs", nargs="+", type=Path, help="Explicit paths to a validation loss CSV file.")

    parser.add_argument(
        "--metric-filename",
        type=str,
        default="validation_loss__total_loss.csv",
        help="Metric CSV filename to load from each run directory (default: validation_loss__total_loss.csv).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels (same order as inputs). If omitted, derived from path.",
    )
    parser.add_argument("--title", type=str, default="Validation loss")
    parser.add_argument("--ylabel", type=str, default="loss")
    parser.add_argument("--cms-subtitle", type=str, default="Internal work")
    parser.add_argument("--max-epoch", type=int, default=25, help="Cap epochs to <= this value (default: 25). Use -1 to disable.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <cwd>/compare_validation_loss.png).",
    )
    args = parser.parse_args()

    max_epoch = None if args.max_epoch is None or args.max_epoch < 0 else args.max_epoch

    run_dirs: List[Path] = []
    metric_filenames: List[str] = []

    if args.runs is not None:
        run_dirs = list(args.runs)
        metric_filenames = [args.metric_filename for _ in run_dirs]
    else:
        for p in args.metric_csvs:
            rd, fn = load_series_from_metric_csv(p, max_epoch=max_epoch)
            run_dirs.append(rd)
            metric_filenames.append(fn)

    series_list: List[Series] = [
        load_series_from_run(rd, metric_filename=fn, max_epoch=max_epoch)
        for rd, fn in zip(run_dirs, metric_filenames)
    ]

    labels: List[str] = []
    if args.labels:
        if len(args.labels) != len(series_list):
            raise SystemExit(f"error: --labels count ({len(args.labels)}) must match inputs ({len(series_list)})")
        labels = list(args.labels)
    else:
        labels = [derive_default_label(d) for d in run_dirs]

    plt, hep = _import_plotting()
    plt.style.use(hep.style.CMS)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for label, s in zip(labels, series_list):
        ax.plot(s.epochs, s.values, marker="o", markersize=3.0, linewidth=1.6, label=label)

    ax.set_title(args.title, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(*auto_ylim_for_losses(series_list))

    ax.legend(
        loc="best",
        frameon=True,
        fontsize=9,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.6,
        handletextpad=0.5,
    )
    add_internal_label(ax, subtitle=args.cms_subtitle)

    fig.tight_layout()

    out = args.out or Path.cwd() / "compare_validation_loss.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

