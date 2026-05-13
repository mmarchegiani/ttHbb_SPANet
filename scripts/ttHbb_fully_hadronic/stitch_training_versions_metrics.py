#!/usr/bin/env python3
"""
Stitch metrics across multiple Lightning/TensorBoard `version_*` folders into one continuous run.

Why:
  Resuming training often creates a NEW logger version (version_4, version_5, ...).
  TensorBoard can show them as separate runs, but if you want a single continuous
  curve (epoch 0..N across resumes), you need to stitch.

What it does:
  - For each provided version directory, finds the newest `events.out.tfevents.*`
  - Extracts a fixed list of scalar tags (same list used by
    `scripts/ttHbb_fully_hadronic/extract_tensorboard_event_metrics.py`)
  - Uses the `epoch` scalar to map step -> epoch
  - Aggregates each metric to ONE value per epoch (mean over steps in that epoch)
  - Concatenates versions in the provided order, shifting epoch numbers so they
    form a single continuous sequence
  - Writes a new `training_metrics/.../stitched_<name>/` directory containing
    `epoch.csv` and one CSV per metric (same format your plotting scripts expect)

Example:
  python scripts/ttHbb_fully_hadronic/stitch_training_versions_metrics.py \
    --versions \
      /home/export/sdurgut/scratch/ttHbb_SPANet/spanet_output/classifier_btag_full/classifier_btag_full/version_4 \
      /home/export/sdurgut/scratch/ttHbb_SPANet/spanet_output/classifier_btag_full/classifier_btag_full/version_5 \
      /home/export/sdurgut/scratch/ttHbb_SPANet/spanet_output/classifier_btag_full/classifier_btag_full/version_6 \
      /home/export/sdurgut/scratch/ttHbb_SPANet/spanet_output/classifier_btag_full/classifier_btag_full/version_7 \
    --out-base /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics \
    --name stitched_v4_v7
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore

        return EventAccumulator
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import TensorBoard's EventAccumulator. Install with:\n"
            "  python -m pip install tensorboard\n"
            f"Original error: {e}"
        ) from e


# Keep in sync with scripts/ttHbb_fully_hadronic/extract_tensorboard_event_metrics.py
SELECTED_TAGS: Sequence[str] = (
    "epoch",
    "CLASSIFICATION/EVENT/signal_accuracy",
    "jet/accuracy_1_of_1",
    "jet/accuracy_1_of_2",
    "jet/accuracy_1_of_3",
    "jet/accuracy_2_of_2",
    "jet/accuracy_2_of_3",
    "jet/accuracy_3_of_3",
    "loss/classification/EVENT/signal",
    "loss/h/assignment_loss",
    "loss/h/detection_loss",
    "loss/t1/assignment_loss",
    "loss/t1/detection_loss",
    "loss/t2/assignment_loss",
    "loss/t2/detection_loss",
    "loss/total_loss",
    "particle/accuracy",
    "particle/accuracy_1_of_1",
    "particle/accuracy_1_of_2",
    "particle/accuracy_1_of_3",
    "particle/accuracy_2_of_2",
    "particle/accuracy_2_of_3",
    "particle/accuracy_3_of_3",
    "particle/f_score",
    "particle/sensitivity",
    "particle/specificity",
    "validation_accuracy",
    "validation_loss/classification/EVENT/signal",
    "validation_loss/h/assignment_loss",
    "validation_loss/h/detection_loss",
    "validation_loss/t1/assignment_loss",
    "validation_loss/t1/detection_loss",
    "validation_loss/t2/assignment_loss",
    "validation_loss/t2/detection_loss",
    "validation_loss/total_loss",
)


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    wall_time: float
    value: float


@dataclass(frozen=True)
class Series:
    epochs: np.ndarray
    values: np.ndarray


def newest_event_file(version_dir: Path) -> Path:
    candidates = sorted(version_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No events.out.tfevents.* found in {version_dir}")
    return candidates[-1]


def load_scalars(event_file: Path, tags: Sequence[str]) -> Dict[str, List[ScalarPoint]]:
    EventAccumulator = _import_event_accumulator()
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()

    out: Dict[str, List[ScalarPoint]] = {}
    for tag in tags:
        try:
            scalars = ea.Scalars(tag)
        except KeyError:
            continue
        out[tag] = [ScalarPoint(step=int(e.step), wall_time=float(e.wall_time), value=float(e.value)) for e in scalars]
    return out


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


def safe_filename(tag: str) -> str:
    return tag.replace("/", "__")


def derive_run_subdir(version_dir: Path) -> Path:
    """
    Convert .../spanet_output/<name>/<name>/version_X -> <name>/<name>/stitched_...
    """
    parts = list(version_dir.resolve().parts)
    if "spanet_output" in parts:
        i = parts.index("spanet_output")
        rel = parts[i + 1 :]
        # drop trailing version_*
        if rel and rel[-1].startswith("version_"):
            rel = rel[:-1]
        if rel:
            return Path(*rel)
    return Path(version_dir.parent.name)


def write_csv(path: Path, epochs: np.ndarray, values: np.ndarray, wall_time: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "wall_time", "value"])
        for e, v in zip(epochs.tolist(), values.tolist()):
            w.writerow([int(e), f"{wall_time:.6f}", f"{float(v):.9g}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch metrics across multiple version_* directories.")
    parser.add_argument("--versions", nargs="+", type=Path, required=True, help="version_* directories to stitch in order")
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("/home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics"),
        help="Base training_metrics directory to write into.",
    )
    parser.add_argument("--name", type=str, default=None, help="Name of stitched folder (default derived from versions).")
    args = parser.parse_args()

    versions = args.versions
    for v in versions:
        if not v.is_dir():
            raise SystemExit(f"error: not a directory: {v}")

    base_subdir = derive_run_subdir(versions[0])
    name = args.name
    if not name:
        vers = [p.name.replace("version_", "v") for p in versions]
        name = "stitched_" + "_".join(vers)

    out_dir = args.out_base / base_subdir / name

    # Load each version's scalars and turn into per-epoch series.
    stitched: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}  # tag -> list of (epochs, values)
    epoch_offset = 0
    global_epoch_list: List[int] = []

    for version_dir in versions:
        ev = newest_event_file(version_dir)
        scalars = load_scalars(ev, SELECTED_TAGS)
        if "epoch" not in scalars or not scalars["epoch"]:
            raise RuntimeError(f"Missing 'epoch' scalar in {ev}")

        epoch_steps = np.asarray([p.step for p in scalars["epoch"]], dtype=np.int64)
        epoch_vals = np.asarray([p.value for p in scalars["epoch"]], dtype=np.float64)
        # Epoch scalar values should be integers.
        epoch_vals_int = epoch_vals.astype(np.int64)

        # Determine how many epochs this version covers.
        local_epoch_max = int(epoch_vals_int.max()) if epoch_vals_int.size else -1
        local_epoch_count = local_epoch_max + 1 if local_epoch_max >= 0 else 0

        # For each tag, build per-epoch series and shift.
        for tag, pts in scalars.items():
            if tag == "epoch":
                continue
            steps = np.asarray([p.step for p in pts], dtype=np.int64)
            vals = np.asarray([p.value for p in pts], dtype=np.float64)
            epochs = steps_to_epochs(steps, epoch_steps, epoch_vals_int)
            s = aggregate_by_epoch(epochs, vals)
            s_shifted = (s.epochs + epoch_offset).astype(np.int64)
            stitched.setdefault(tag, []).append((s_shifted, s.values))

        # Add global epoch coverage for epoch.csv
        global_epoch_list.extend(list(range(epoch_offset, epoch_offset + local_epoch_count)))
        epoch_offset += local_epoch_count

    # Write epoch.csv as identity map (step==epoch, value==epoch).
    global_epochs = np.asarray(sorted(set(global_epoch_list)), dtype=np.int64)
    write_csv(out_dir / "epoch.csv", global_epochs, global_epochs.astype(np.float64))

    # Write one CSV per stitched tag.
    for tag, parts in stitched.items():
        # Concatenate parts (already shifted to unique epoch ranges)
        epochs = np.concatenate([p[0] for p in parts]).astype(np.int64)
        values = np.concatenate([p[1] for p in parts]).astype(np.float64)
        order = np.argsort(epochs)
        epochs = epochs[order]
        values = values[order]
        write_csv(out_dir / f"{safe_filename(tag)}.csv", epochs, values)

    print(f"Wrote stitched metrics to: {out_dir}")


if __name__ == "__main__":
    main()

