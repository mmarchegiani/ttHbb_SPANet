#!/usr/bin/env python3
"""
Plot extracted training metrics (CSV files) vs epoch using CMS-style plotting (mplhep).

Input directory is produced by:
  scripts/extract_tensorboard_event_metrics.py

Expected layout:
  <metrics_dir>/
    epoch.csv
    validation_loss__total_loss.csv
    loss__total_loss.csv
    ...

This script:
  - uses epoch.csv (step -> epoch) to map each metric point to an epoch
  - aggregates metrics to one value per epoch (mean over steps within epoch)
  - writes one plot per metric under <metrics_dir>/plots/

Example:
  python scripts/plot_training_metrics_mplhep.py \
    /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_T/classifier_btag_T/version_1
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    epochs: np.ndarray  # shape (N,)
    values: np.ndarray  # shape (N,)


def cap_series(series: Series, *, max_epoch: int | None) -> Series:
    if max_epoch is None:
        return series
    m = series.epochs <= max_epoch
    return Series(epochs=series.epochs[m], values=series.values[m])


def read_metric_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps: List[int] = []
    wall: List[float] = []
    vals: List[float] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            wall.append(float(row["wall_time"]))
            vals.append(float(row["value"]))
    return np.asarray(steps, dtype=np.int64), np.asarray(wall, dtype=np.float64), np.asarray(vals, dtype=np.float64)


def steps_to_epochs(metric_steps: np.ndarray, epoch_steps: np.ndarray, epoch_values: np.ndarray) -> np.ndarray:
    """
    Map each metric step to the most recent epoch value at or before that step.
    """
    # indices of rightmost epoch_step <= metric_step
    idx = np.searchsorted(epoch_steps, metric_steps, side="right") - 1
    idx = np.clip(idx, 0, len(epoch_steps) - 1)
    return epoch_values[idx].astype(np.int64)


def aggregate_by_epoch(epochs: np.ndarray, values: np.ndarray) -> Series:
    """
    Reduce to one value per epoch by averaging all points that fall into that epoch.
    """
    order = np.argsort(epochs)
    epochs = epochs[order]
    values = values[order]

    uniq = np.unique(epochs)
    out_vals = np.empty_like(uniq, dtype=np.float64)
    for i, e in enumerate(uniq):
        m = epochs == e
        out_vals[i] = float(values[m].mean())
    return Series(epochs=uniq, values=out_vals)


def metric_label_from_filename(file: Path) -> str:
    # reverse of safe_filename: "__" -> "/"
    return file.stem.replace("__", "/")


def is_loss(tag: str) -> bool:
    return "loss" in tag.lower()


def is_accuracy_like(tag: str) -> bool:
    t = tag.lower()
    return ("accuracy" in t) or ("f_score" in t) or ("sensitivity" in t) or ("specificity" in t)


def best_value(tag: str, y: np.ndarray) -> float:
    # losses: lower is better, otherwise higher is better
    return float(np.nanmin(y) if is_loss(tag) else np.nanmax(y))


def add_internal_label(ax, *, subtitle: str = "Internal work") -> None:
    """
    Replace mplhep's default 'Simulation' header with a compact, custom label.
    """
    # Coordinates are axes-fraction.
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


def plot_one(tag: str, series: Series, out_path: Path, *, cms_label: str = "CMS") -> None:
    plt, hep = _import_plotting()

    plt.style.use(hep.style.CMS)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(series.epochs, series.values, marker="o", markersize=3.2, linewidth=1.4, label=tag)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(tag)
    ax.grid(True, which="both", alpha=0.25)

    last = float(series.values[-1]) if series.values.size else float("nan")
    best = best_value(tag, series.values) if series.values.size else float("nan")

    # Compact legend + a small stats box (keeps legend readable)
    ax.legend(
        loc="best",
        frameon=True,
        fontsize=9,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.6,
        handletextpad=0.5,
    )
    ax.text(
        0.98,
        0.02,
        f"last = {last:.6g}\nbest = {best:.6g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    # Custom CMS label (no Simulation / no energy header)
    add_internal_label(ax, subtitle=cms_label)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_overlay(
    ax,
    *,
    title: str,
    series_by_label: Dict[str, Series],
    ylabel: str,
    cms_label: str,
):
    plt, hep = _import_plotting()
    plt.style.use(hep.style.CMS)

    for label, s in series_by_label.items():
        if s.epochs.size == 0:
            continue
        ax.plot(s.epochs, s.values, marker="o", markersize=3.0, linewidth=1.4, label=label)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(
        loc="best",
        frameon=True,
        fontsize=8,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=1.4,
        handletextpad=0.45,
        ncol=1,
    )
    add_internal_label(ax, subtitle=cms_label)


def write_summary_pdf(
    *,
    out_pdf: Path,
    run_name: str,
    series_by_tag: Dict[str, Series],
    cms_label: str,
    max_epoch: int | None,
) -> None:
    """
    Multi-page PDF summary of key metrics.
    """
    plt, hep = _import_plotting()
    from matplotlib.backends.backend_pdf import PdfPages  # type: ignore

    plt.style.use(hep.style.CMS)

    def get(tag: str) -> Series | None:
        s = series_by_tag.get(tag)
        return None if s is None else cap_series(s, max_epoch=max_epoch)

    def short_run_id(name: str) -> str:
        # Try to extract something like "btag_t" from "classifier_btag_T/.../version_1"
        lowered = name.lower()
        if "classifier_" in lowered:
            after = lowered.split("classifier_", 1)[1]
            token = after.split("/", 1)[0]
            return token
        # Fallback: last folder-ish token
        return lowered.strip("/").split("/")[-1]

    run_id = short_run_id(run_name)
    header = f"{run_id} — {run_name}"

    with PdfPages(out_pdf) as pdf:
        # Page 1: total loss (train vs validation)
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        overlays: Dict[str, Series] = {}
        if get("loss/total_loss") is not None:
            overlays["train loss/total_loss"] = get("loss/total_loss")  # type: ignore[assignment]
        if get("validation_loss/total_loss") is not None:
            overlays["val loss/total_loss"] = get("validation_loss/total_loss")  # type: ignore[assignment]
        _plot_overlay(
            ax,
            title=f"{header} — total loss",
            series_by_label=overlays,
            ylabel="loss",
            cms_label=cms_label,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: assignment/detection losses by particle (validation)
        fig, axs = plt.subplots(2, 3, figsize=(11, 7), sharex=True)
        parts = ["h", "t1", "t2"]
        for j, p in enumerate(parts):
            for i, kind in enumerate(["assignment_loss", "detection_loss"]):
                ax = axs[i, j]
                overlays = {}
                tag_val = f"validation_loss/{p}/{kind}"
                tag_tr = f"loss/{p}/{kind}"
                if get(tag_tr) is not None:
                    overlays[f"train {p}/{kind}"] = get(tag_tr)  # type: ignore[assignment]
                if get(tag_val) is not None:
                    overlays[f"val {p}/{kind}"] = get(tag_val)  # type: ignore[assignment]
                _plot_overlay(
                    ax,
                    title=f"{p} {kind}",
                    series_by_label=overlays,
                    ylabel="loss",
                    cms_label=cms_label,
                )
        fig.suptitle(f"{header} — assignment/detection losses", y=0.985, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: accuracies (validation + jet/particle)
        fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        acc_pages = [
            ("validation_accuracy", "validation_accuracy"),
            ("CLASSIFICATION/EVENT/signal_accuracy", "signal classification accuracy"),
            ("particle/accuracy", "particle accuracy"),
            ("jet/accuracy_1_of_1", "jet accuracy 1-of-1"),
        ]
        for ax, (tag, title) in zip(axs.ravel(), acc_pages):
            overlays = {}
            if get(tag) is not None:
                overlays[tag] = get(tag)  # type: ignore[assignment]
            _plot_overlay(ax, title=title, series_by_label=overlays, ylabel="accuracy", cms_label=cms_label)
            ax.set_ylim(0.0, 1.05)
        fig.suptitle(f"{header} — key accuracies", y=0.985, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: jet and particle breakdowns
        fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        overlays = {
            "jet 1/1": get("jet/accuracy_1_of_1"),
            "jet 1/2": get("jet/accuracy_1_of_2"),
            "jet 1/3": get("jet/accuracy_1_of_3"),
            "jet 2/2": get("jet/accuracy_2_of_2"),
            "jet 2/3": get("jet/accuracy_2_of_3"),
            "jet 3/3": get("jet/accuracy_3_of_3"),
        }
        _plot_overlay(
            axs[0, 0],
            title="jet accuracies",
            series_by_label={k: v for k, v in overlays.items() if v is not None},
            ylabel="accuracy",
            cms_label=cms_label,
        )
        axs[0, 0].set_ylim(0.0, 1.05)

        overlays = {
            "particle 1/1": get("particle/accuracy_1_of_1"),
            "particle 1/2": get("particle/accuracy_1_of_2"),
            "particle 1/3": get("particle/accuracy_1_of_3"),
            "particle 2/2": get("particle/accuracy_2_of_2"),
            "particle 2/3": get("particle/accuracy_2_of_3"),
            "particle 3/3": get("particle/accuracy_3_of_3"),
        }
        _plot_overlay(
            axs[0, 1],
            title="particle accuracies",
            series_by_label={k: v for k, v in overlays.items() if v is not None},
            ylabel="accuracy",
            cms_label=cms_label,
        )
        axs[0, 1].set_ylim(0.0, 1.05)

        overlays = {
            "particle/f_score": get("particle/f_score"),
            "particle/sensitivity": get("particle/sensitivity"),
            "particle/specificity": get("particle/specificity"),
        }
        _plot_overlay(
            axs[1, 0],
            title="particle quality metrics",
            series_by_label={k: v for k, v in overlays.items() if v is not None},
            ylabel="score",
            cms_label=cms_label,
        )
        axs[1, 0].set_ylim(0.0, 1.05)

        overlays = {
            "val class loss": get("validation_loss/classification/EVENT/signal"),
            "train class loss": get("loss/classification/EVENT/signal"),
        }
        _plot_overlay(
            axs[1, 1],
            title="classification loss (signal)",
            series_by_label={k: v for k, v in overlays.items() if v is not None},
            ylabel="loss",
            cms_label=cms_label,
        )

        fig.suptitle(f"{header} — breakdown", y=0.985, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot extracted training metrics per epoch (mplhep CMS style).")
    parser.add_argument("metrics_dir", type=Path, help="Directory containing epoch.csv and metric CSVs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <metrics_dir>/plots).",
    )
    parser.add_argument(
        "--cms-label",
        type=str,
        default="CMS",
        help="CMS label text passed to mplhep (default: CMS).",
    )
    parser.add_argument(
        "--no-summary-pdf",
        action="store_true",
        help="Disable writing a multi-page summary PDF (summary.pdf).",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=25,
        help="Cap plots to epochs <= this value (default: 25). Use -1 to disable.",
    )
    args = parser.parse_args()

    metrics_dir: Path = args.metrics_dir
    if not metrics_dir.is_dir():
        raise SystemExit(f"error: not a directory: {metrics_dir}")

    epoch_csv = metrics_dir / "epoch.csv"
    if not epoch_csv.is_file():
        raise SystemExit(f"error: missing epoch.csv in {metrics_dir}")

    epoch_steps, _epoch_wall, epoch_vals = read_metric_csv(epoch_csv)

    out_dir = args.out_dir or (metrics_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot all metric CSVs except epoch.csv
    csv_files = sorted(p for p in metrics_dir.glob("*.csv") if p.name != "epoch.csv")
    if not csv_files:
        raise SystemExit(f"error: no metric CSVs found in {metrics_dir}")

    series_by_tag: Dict[str, Series] = {}
    for csv_path in csv_files:
        tag = metric_label_from_filename(csv_path)
        steps, _wall, values = read_metric_csv(csv_path)

        epochs = steps_to_epochs(steps, epoch_steps, epoch_vals)
        series = aggregate_by_epoch(epochs, values)
        max_epoch = None if args.max_epoch is None or args.max_epoch < 0 else args.max_epoch
        series = cap_series(series, max_epoch=max_epoch)

        # Choose y-axis label/formatting via tag; filename is tag-safe
        plot_name = csv_path.stem + ".png"
        plot_one(tag, series, out_dir / plot_name, cms_label=args.cms_label)
        series_by_tag[tag] = series

    if not args.no_summary_pdf:
        run_name = str(metrics_dir).split("training_metrics/")[-1].rstrip("/")
        write_summary_pdf(
            out_pdf=out_dir / "summary.pdf",
            run_name=run_name,
            series_by_tag=series_by_tag,
            cms_label=args.cms_label,
            max_epoch=max_epoch,
        )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()

