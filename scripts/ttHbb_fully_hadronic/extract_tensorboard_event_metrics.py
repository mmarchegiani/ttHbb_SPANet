#!/usr/bin/env python3
"""
Extract metrics from a single TensorBoard event file (events.out.tfevents.*).

This is useful when you want to plot metrics without running TensorBoard, or
to export scalars into JSON/CSV for custom plotting.

Example:
  python scripts/extract_tensorboard_event_metrics.py \
    /home/export/sdurgut/scratch/ttHbb_SPANet/spanet_output/classifier_btag_T/classifier_btag_T/version_1/events.out.tfevents.1775008364.rogue01 \
    --out-json metrics.json --out-csv metrics.csv

Notes:
  - This script requires the `tensorboard` Python package.
    If missing in your env: `python -m pip install tensorboard`
  - It extracts:
      * scalar time series (step, wall_time, value)
      * (optionally) hparams / metadata tags listing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore

        return EventAccumulator
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import TensorBoard's EventAccumulator. "
            "Install it in this environment with: `python -m pip install tensorboard`.\n"
            f"Original error: {e}"
        ) from e


@dataclass(frozen=True)
class ScalarPoint:
    tag: str
    step: int
    wall_time: float
    value: float


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


def extract_scalars(
    event_file: Path,
    *,
    selected_tags: Sequence[str],
    size_guidance: Optional[Dict[str, int]] = None,
) -> List[ScalarPoint]:
    EventAccumulator = _import_event_accumulator()

    # Default guidance: keep everything. Users can cap later if files are massive.
    if size_guidance is None:
        size_guidance = {
            "scalars": 0,
            "images": 0,
            "audio": 0,
            "histograms": 0,
            "tensors": 0,
            "compressedHistograms": 0,
        }

    ea = EventAccumulator(str(event_file), size_guidance=size_guidance)
    ea.Reload()

    points: List[ScalarPoint] = []
    for tag in selected_tags:
        try:
            series = ea.Scalars(tag)
        except KeyError:
            continue
        for e in series:
            # e has fields: wall_time, step, value
            points.append(ScalarPoint(tag=tag, step=int(e.step), wall_time=float(e.wall_time), value=float(e.value)))

    # Stable ordering for reproducible outputs
    points.sort(key=lambda p: (p.tag, p.step, p.wall_time))
    return points


def summarize(points: Iterable[ScalarPoint]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for p in points:
        s = summary.setdefault(
            p.tag,
            {"count": 0, "min_step": None, "max_step": None, "min": None, "max": None},
        )
        s["count"] += 1
        s["min_step"] = p.step if s["min_step"] is None else min(s["min_step"], p.step)
        s["max_step"] = p.step if s["max_step"] is None else max(s["max_step"], p.step)
        s["min"] = p.value if s["min"] is None else min(s["min"], p.value)
        s["max"] = p.value if s["max"] is None else max(s["max"], p.value)
    return summary


def write_json(out_path: Path, *, event_file: Path, points: List[ScalarPoint]) -> None:
    payload = {
        "event_file": str(event_file),
        "scalars": [
            {"tag": p.tag, "step": p.step, "wall_time": p.wall_time, "value": p.value} for p in points
        ],
        "summary": summarize(points),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def write_csv(out_path: Path, points: List[ScalarPoint]) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "wall_time", "value"])
        for p in points:
            w.writerow([p.tag, p.step, f"{p.wall_time:.6f}", f"{p.value:.9g}"])


def safe_filename(tag: str) -> str:
    return tag.replace("/", "__")


def derive_run_subdir(event_file: Path) -> Path:
    parts = list(event_file.resolve().parts)
    if "spanet_output" in parts:
        i = parts.index("spanet_output")
        rel_parts = parts[i + 1 : -1]  # after spanet_output, up to parent dir
        if rel_parts:
            return Path(*rel_parts)
    return Path(event_file.parent.name)


def write_selected_metrics_tree(
    *,
    event_file: Path,
    out_base: Path,
    selected_tags: Sequence[str],
    print_missing: bool,
) -> Path:
    run_subdir = derive_run_subdir(event_file)
    out_dir = out_base / run_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    points = extract_scalars(event_file, selected_tags=selected_tags)
    present = {p.tag for p in points}
    missing = [t for t in selected_tags if t not in present]
    if print_missing and missing:
        print("Missing tags (not found in this event file):")
        for t in missing:
            print(f"  {t}")

    by_tag: Dict[str, List[ScalarPoint]] = {}
    for p in points:
        by_tag.setdefault(p.tag, []).append(p)

    for tag, series in by_tag.items():
        out_csv = out_dir / f"{safe_filename(tag)}.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "wall_time", "value"])
            for p in series:
                w.writerow([p.step, f"{p.wall_time:.6f}", f"{p.value:.9g}"])

    write_json(out_dir / "metrics.json", event_file=event_file, points=points)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected scalar metrics from a TensorBoard event file into per-metric CSVs."
    )
    parser.add_argument("event_file", type=Path, help="Path to a single events.out.tfevents.* file.")
    parser.add_argument("--list-tags", action="store_true", help="Print available scalar tags and exit.")
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("/home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics"),
        help="Base output directory where the run subdirectory will be created.",
    )
    parser.add_argument(
        "--print-missing",
        action="store_true",
        help="Print which requested metrics were not present in the event file.",
    )
    args = parser.parse_args()

    if not args.event_file.is_file():
        print(f"error: not a file: {args.event_file}", file=sys.stderr)
        sys.exit(2)

    # Load tags first so --list-tags can be fast-ish.
    EventAccumulator = _import_event_accumulator()
    ea = EventAccumulator(str(args.event_file))
    ea.Reload()
    tags = ea.Tags()
    scalar_tags = tags.get("scalars", []) or []

    if args.list_tags:
        for t in sorted(scalar_tags):
            print(t)
        return

    out_dir = write_selected_metrics_tree(
        event_file=args.event_file,
        out_base=args.out_base,
        selected_tags=SELECTED_TAGS,
        print_missing=args.print_missing,
    )
    print(f"Wrote metrics to: {out_dir}")


if __name__ == "__main__":
    main()

