#!/usr/bin/env python3
"""
Overlay CLASSIFICATION/EVENT/signal_accuracy curves from multiple training runs.

Inputs are the extracted-metrics CSVs produced under:
  /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/<run>/.../version_X/

For each run, we use:
  - epoch.csv to map step -> epoch
  - CLASSIFICATION__EVENT__signal_accuracy.csv for the metric values

Example:
  python scripts/ttHbb_fully_hadronic/compare_signal_accuracy_runs.py \
    --runs \
      /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_T/classifier_btag_T/version_1 \
      /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_TM/classifier_btag_TM/version_4 \
    --labels btag_T btag_TM \
    --title "Signal classification accuracy" \
    --cms-subtitle "Internal work" \
    --max-epoch 25 \
    --out /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/compare_signal_accuracy.png

You can also pass explicit metric CSV paths with --metric-csvs.
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


METRIC_FILENAME = "CLASSIFICATION__EVENT__signal_accuracy.csv"
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
    # Prefer something compact like "classifier_btag_TM/version_4"
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


def auto_ylim_for_series(series_list: List[Series]) -> tuple[float, float]:
    """
    Choose a tight y-limits window around the data (useful when curves are ~1.0).
    Always clamps to [0, 1.05].
    """
    ys: List[np.ndarray] = [s.values for s in series_list if s.values.size]
    if not ys:
        return 0.0, 1.05

    y = np.concatenate(ys)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.05

    # Robust bounds to avoid single-point outliers dominating the view.
    lo = float(np.percentile(y, 1))
    hi = float(np.percentile(y, 99))
    span = max(hi - lo, 1e-6)

    # Pad a bit so lines don't touch the frame.
    pad = max(0.02 * span, 0.002)
    lo2 = lo - pad
    hi2 = hi + pad

    # If everything is extremely close, zoom to a ~1% window.
    if (hi2 - lo2) < 0.01:
        center = 0.5 * (hi2 + lo2)
        lo2 = center - 0.005
        hi2 = center + 0.005

    lo2 = max(0.0, lo2)
    hi2 = min(1.05, hi2)
    if hi2 <= lo2:
        return 0.0, 1.05
    return lo2, hi2


def load_series_from_run(run_dir: Path, *, max_epoch: int | None) -> Series:
    epoch_csv = run_dir / EPOCH_FILENAME
    metric_csv = run_dir / METRIC_FILENAME
    if not epoch_csv.is_file():
        raise FileNotFoundError(f"Missing {EPOCH_FILENAME} in {run_dir}")
    if not metric_csv.is_file():
        raise FileNotFoundError(f"Missing {METRIC_FILENAME} in {run_dir}")

    epoch_steps, epoch_vals = read_metric_csv(epoch_csv)
    metric_steps, metric_vals = read_metric_csv(metric_csv)

    epochs = steps_to_epochs(metric_steps, epoch_steps, epoch_vals)
    series = aggregate_by_epoch(epochs, metric_vals)
    return cap_series(series, max_epoch=max_epoch)


def load_series_from_metric_csv(metric_csv: Path, *, max_epoch: int | None) -> tuple[Path, Series]:
    run_dir = metric_csv.parent
    if metric_csv.name != METRIC_FILENAME:
        raise ValueError(f"Expected metric csv named {METRIC_FILENAME}, got {metric_csv}")
    return run_dir, load_series_from_run(run_dir, max_epoch=max_epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay signal accuracy curves from multiple runs (CMS style).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", nargs="+", type=Path, help="Run directories containing epoch.csv and metric CSV.")
    group.add_argument("--metric-csvs", nargs="+", type=Path, help=f"Explicit paths to {METRIC_FILENAME} files.")

    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels (same order as inputs). If omitted, derived from path.",
    )
    parser.add_argument("--title", type=str, default="Signal classification accuracy")
    parser.add_argument("--ylabel", type=str, default="accuracy")
    parser.add_argument("--cms-subtitle", type=str, default="Internal work")
    parser.add_argument("--max-epoch", type=int, default=25, help="Cap epochs to <= this value (default: 25). Use -1 to disable.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <cwd>/compare_signal_accuracy.png).",
    )
    args = parser.parse_args()

    max_epoch = None if args.max_epoch is None or args.max_epoch < 0 else args.max_epoch

    run_dirs: List[Path] = []
    series_list: List[Series] = []

    if args.runs is not None:
        for rd in args.runs:
            run_dirs.append(rd)
            series_list.append(load_series_from_run(rd, max_epoch=max_epoch))
    else:
        for p in args.metric_csvs:
            rd, s = load_series_from_metric_csv(p, max_epoch=max_epoch)
            run_dirs.append(rd)
            series_list.append(s)

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
    ax.set_ylim(*auto_ylim_for_series(series_list))

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

    out = args.out or Path.cwd() / "compare_signal_accuracy.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

