#!/usr/bin/env python3
"""
Jet multiplicity inspection suite (parquet + H5), per chunk + overall.

Goal: quantify how often capping jets to 16 drops jets, using:
- Parquet JetGood (raw jagged multiplicity)
- Parquet JetGoodMatched (if present)
- H5 INPUTS/Jet/MASK (post-cap multiplicity)

Also computes for signal_only:
- matched-jet multiplicity (sum(JetGood.matched))
- clipping risk: fraction of events with any matched/prov jet at original local index >= 16

Outputs per chunk and overall:
- PNG plots (log-y + linear-y) with descriptive titles and embedded stats, plus a vertical line at 16
- CSV histogram bin counts (integer jet multiplicities)
- JSON stats summaries

Optional EOS upload:
- Uploads the entire output directory tree to an EOS path via xrdfs mkdir -p + xrdcp -f
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


CHUNKS = {
    "signal_only": [
        "output_TTH_Hto2B_2022_postEE.parquet",
        "output_TTH_Hto2B_2022_preEE.parquet",
        "output_TTH_Hto2B_2023_postBPix.parquet",
        "output_TTH_Hto2B_2023_preBPix.parquet",
    ],
    "qcd_ht200_600": [
        "output_QCD-4Jets_HT-200to400_*.parquet",
        "output_QCD-4Jets_HT-400to600_*.parquet",
    ],
    "qcd_ht600_1000": [
        "output_QCD-4Jets_HT-600to800_*.parquet",
        "output_QCD-4Jets_HT-800to1000_*.parquet",
    ],
    "qcd_ht1000_1500": [
        "output_QCD-4Jets_HT-1000to1200_*.parquet",
        "output_QCD-4Jets_HT-1200to1500_*.parquet",
    ],
    "qcd_ht1500_2000": [
        "output_QCD-4Jets_HT-1500to2000_*.parquet",
    ],
    "qcd_ht2000": [
        "output_QCD-4Jets_HT-2000_*.parquet",
    ],
}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class StreamingStats:
    max_bin: int
    n: int = 0
    sum: int = 0
    min: int = 10**18
    max: int = -10**18
    ge16: int = 0
    gt16: int = 0
    overflow: int = 0

    def __post_init__(self) -> None:
        self.hist = np.zeros(self.max_bin + 1, dtype=np.int64)  # bins 0..max_bin inclusive

    def update(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.int64)
        if v.size == 0:
            return

        self.n += int(v.shape[0])
        self.sum += int(v.sum())
        self.min = int(min(self.min, int(v.min())))
        self.max = int(max(self.max, int(v.max())))
        self.ge16 += int(np.sum(v >= 16))
        self.gt16 += int(np.sum(v > 16))

        in_range = v[(v >= 0) & (v <= self.max_bin)]
        if in_range.size:
            self.hist += np.bincount(in_range, minlength=self.max_bin + 1).astype(np.int64)
        self.overflow += int(np.sum(v > self.max_bin))

    def mean(self) -> float:
        return float(self.sum / self.n) if self.n else float("nan")

    def frac_ge16(self) -> float:
        return float(self.ge16 / self.n) if self.n else float("nan")

    def frac_gt16(self) -> float:
        return float(self.gt16 / self.n) if self.n else float("nan")

    def approx_percentile(self, q: float) -> float:
        if not self.n:
            return float("nan")
        if q <= 0:
            return float(self.min)
        if q >= 1:
            return float(self.max)
        target = q * (self.n - 1)
        cum = 0
        for k, c in enumerate(self.hist):
            prev = cum
            cum += int(c)
            if cum > target:
                if c == 0:
                    return float(k)
                # return integer bin center k (multiplicity is discrete)
                _ = (target - prev) / c
                return float(k)
        return float(max(self.max_bin, self.max))

    def to_dict(self) -> dict:
        return {
            "n_events": int(self.n),
            "min": int(self.min) if self.n else None,
            "max": int(self.max) if self.n else None,
            "mean": float(self.mean()) if self.n else None,
            "p50": float(self.approx_percentile(0.50)) if self.n else None,
            "p90": float(self.approx_percentile(0.90)) if self.n else None,
            "p95": float(self.approx_percentile(0.95)) if self.n else None,
            "p99": float(self.approx_percentile(0.99)) if self.n else None,
            "frac_ge16": float(self.frac_ge16()) if self.n else None,
            "frac_gt16": float(self.frac_gt16()) if self.n else None,
            "overflow_gt_max_bin": int(self.overflow),
            "max_bin": int(self.max_bin),
        }

    def write_hist_csv(self, out_csv: Path) -> None:
        out_csv.write_text(
            "multiplicity,count\n" + "\n".join(f"{i},{int(c)}" for i, c in enumerate(self.hist)),
            encoding="utf-8",
        )


def plot_from_hist(out_png: Path, title: str, subtitle: str, stats: StreamingStats, logy: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib not available; skipping {out_png.name}. ({exc})")
        return

    x = np.arange(0, stats.max_bin + 1)
    y = stats.hist

    plt.figure(figsize=(10.2, 6.6))
    plt.step(x, y, where="mid", linewidth=2)
    plt.axvline(16, linestyle="--", linewidth=1.5)
    plt.xlabel("Jets per event")
    plt.ylabel("Events")
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.text(0.01, 0.98, subtitle, transform=plt.gca().transAxes, va="top", ha="left", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def expand_patterns(base: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(base.glob(pat)))
    return out


def parquet_stream_counts(
    parquet_files: list[Path],
    collection: str,
    max_bin: int,
    pt_sorted_check_events: int = 0,
) -> tuple[StreamingStats, Optional[dict]]:
    import awkward as ak

    stats = StreamingStats(max_bin=max_bin)
    ordering_report = None

    checked = 0
    violations = 0
    total_checked = 0

    for p in parquet_files:
        arr = ak.from_parquet(str(p))
        if collection not in arr.fields:
            raise KeyError(f"{p}: missing '{collection}', available: {arr.fields}")
        jets = arr[collection]
        counts = ak.num(jets, axis=1)
        stats.update(np.asarray(counts, dtype=np.int64))

        if pt_sorted_check_events > 0 and collection == "JetGood" and "pt" in jets.fields and checked < pt_sorted_check_events:
            n_take = min(int(pt_sorted_check_events - checked), int(len(jets)))
            pt = jets.pt[:n_take]
            ok = ak.all(pt[:, :-1] >= pt[:, 1:], axis=1)
            ok_np = np.asarray(ok, dtype=bool)
            total_checked += int(ok_np.shape[0])
            violations += int((~ok_np).sum())
            checked += int(ok_np.shape[0])

    if pt_sorted_check_events > 0 and total_checked > 0:
        ordering_report = {
            "checked_events": int(total_checked),
            "unsorted_events": int(violations),
            "frac_unsorted": float(violations / total_checked),
        }

    return stats, ordering_report


def signal_clipping_risk(parquet_files: list[Path], collection: str = "JetGood", max_index: int = 15) -> dict:
    import awkward as ak

    any_matched = []
    any_prov = []

    for p in parquet_files:
        arr = ak.from_parquet(str(p))
        jets = arr[collection]
        idx = ak.local_index(jets)

        if "matched" not in jets.fields:
            raise KeyError(f"{p}: {collection}.matched missing")
        m = jets.matched
        any_matched.append(ak.any(m & (idx > max_index), axis=1))

        if "prov" in jets.fields:
            prov_ok = jets.prov >= 0
            any_prov.append(ak.any(prov_ok & (idx > max_index), axis=1))

    any_matched_np = np.asarray(ak.concatenate(any_matched, axis=0), dtype=bool)
    payload = {
        "max_index": int(max_index),
        "frac_any_matched_idx_gt_max": float(any_matched_np.mean()),
    }
    if any_prov:
        any_prov_np = np.asarray(ak.concatenate(any_prov, axis=0), dtype=bool)
        payload["frac_any_prov_idx_gt_max"] = float(any_prov_np.mean())
    return payload


def parquet_stream_matched_counts(parquet_files: list[Path], max_bin: int) -> StreamingStats:
    import awkward as ak

    stats = StreamingStats(max_bin=max_bin)
    for p in parquet_files:
        arr = ak.from_parquet(str(p))
        jets = arr["JetGood"]
        if "matched" not in jets.fields:
            raise KeyError(f"{p}: JetGood.matched missing")
        mcounts = ak.sum(jets.matched, axis=1)
        stats.update(np.asarray(mcounts, dtype=np.int64))
    return stats


def h5_stream_mask_counts(h5_files: list[Path], max_bin: int, chunk_rows: int = 250_000) -> StreamingStats:
    import h5py

    stats = StreamingStats(max_bin=max_bin)
    for p in h5_files:
        with h5py.File(str(p), "r") as f:
            ds = f["INPUTS/Jet/MASK"]
            n = ds.shape[0]
            for start in range(0, n, chunk_rows):
                end = min(n, start + chunk_rows)
                mask = np.asarray(ds[start:end]).astype(bool)
                stats.update(mask.sum(axis=1).astype(np.int64))
    return stats


def _eos_split(eos_url: str) -> tuple[str, str]:
    if not eos_url.startswith("root://"):
        raise ValueError(f"EOS URL must start with root://, got: {eos_url}")
    rest = eos_url[len("root://") :]
    host, path = rest.split("//", 1)
    return host, f"/{path}"


def eos_mkdir_p(eos_dir: str) -> None:
    host, path = _eos_split(eos_dir)
    subprocess.run(["xrdfs", host, "mkdir", "-p", path], check=True)


def eos_upload_tree(out_dir: Path, eos_dir: str) -> None:
    eos_mkdir_p(eos_dir)
    for p in sorted(out_dir.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(out_dir).as_posix()
        target = eos_dir.rstrip("/") + "/" + rel
        eos_mkdir_p("/".join(target.split("/")[:-1]))
        subprocess.run(["xrdcp", "-f", str(p), target], check=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_base", default="/home/export/sdurgut/scratch/coffea_to_parquet_output")
    ap.add_argument("--h5_base", default="/home/export/sdurgut/scratch/parquet_to_h5_output/classification/with_MASK")
    ap.add_argument("--run_tag", default="0324")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_bin", type=int, default=40)
    ap.add_argument("--pt_sorted_check_events", type=int, default=200_000)
    ap.add_argument("--eos_dir", default=None)
    ap.add_argument("--skip_eos", action="store_true")
    ap.add_argument("--skip_parquet", action="store_true", help="Skip parquet scanning (useful if awkward is unavailable)")
    ap.add_argument("--skip_h5", action="store_true", help="Skip H5 scanning")
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir) if args.out_dir else (project_dir / "outputs" / f"jet_mult_{args.run_tag}_{now_tag()}")
    ensure_dir(out_dir)

    parquet_base = Path(args.parquet_base)
    h5_base = Path(args.h5_base)

    summary = {
        "run_tag": args.run_tag,
        "created_at": datetime.now().isoformat(),
        "parquet_base": str(parquet_base),
        "h5_base": str(h5_base),
        "max_bin": int(args.max_bin),
        "chunks": {},
    }

    all_parquet: list[Path] = []

    for chunk, patterns in CHUNKS.items():
        chunk_dir = out_dir / chunk
        ensure_dir(chunk_dir)

        pq_files: list[Path] = []
        jetgood_payload = None
        jetgoodmatched_payload = None
        signal_extras = None

        if not args.skip_parquet:
            pq_files = expand_patterns(parquet_base, patterns)
            if not pq_files:
                raise SystemExit(f"No parquet files found for chunk {chunk} under {parquet_base}")
            all_parquet.extend(pq_files)

            # Parquet JetGood multiplicity
            pt_check = args.pt_sorted_check_events if chunk in {"signal_only", "qcd_ht200_600"} else 0
            jetgood_stats, ordering = parquet_stream_counts(
                pq_files, collection="JetGood", max_bin=args.max_bin, pt_sorted_check_events=pt_check
            )
            jetgood_payload = jetgood_stats.to_dict()
            jetgood_payload["num_parquet_files"] = int(len(pq_files))
            if ordering:
                jetgood_payload["pt_sorted_check"] = ordering
            write_json(chunk_dir / "parquet_JetGood_stats.json", jetgood_payload)
            jetgood_stats.write_hist_csv(chunk_dir / "parquet_JetGood_hist.csv")
            subtitle = (
                f"files={len(pq_files)}  N={jetgood_payload['n_events']:,}  "
                f"P(N>16)={jetgood_payload['frac_gt16']:.4%}  "
                f"min={jetgood_payload['min']} max={jetgood_payload['max']}  "
                f"p90≈{jetgood_payload['p90']:.0f} p99≈{jetgood_payload['p99']:.0f}"
            )
            plot_from_hist(
                chunk_dir / f"parquet_JetGood_{chunk}_logy.png",
                f"Jet multiplicity (parquet) — {chunk} — JetGood",
                subtitle,
                jetgood_stats,
                logy=True,
            )
            plot_from_hist(
                chunk_dir / f"parquet_JetGood_{chunk}_lineary.png",
                f"Jet multiplicity (parquet) — {chunk} — JetGood",
                subtitle,
                jetgood_stats,
                logy=False,
            )

            # Parquet JetGoodMatched multiplicity (if present)
            try:
                jgm_stats, _ = parquet_stream_counts(
                    pq_files, collection="JetGoodMatched", max_bin=args.max_bin, pt_sorted_check_events=0
                )
                jetgoodmatched_payload = jgm_stats.to_dict()
                jetgoodmatched_payload["num_parquet_files"] = int(len(pq_files))
                write_json(chunk_dir / "parquet_JetGoodMatched_stats.json", jetgoodmatched_payload)
                jgm_stats.write_hist_csv(chunk_dir / "parquet_JetGoodMatched_hist.csv")
                subtitle2 = (
                    f"files={len(pq_files)}  N={jetgoodmatched_payload['n_events']:,}  "
                    f"P(N>16)={jetgoodmatched_payload['frac_gt16']:.4%}  "
                    f"min={jetgoodmatched_payload['min']} max={jetgoodmatched_payload['max']}"
                )
                plot_from_hist(
                    chunk_dir / f"parquet_JetGoodMatched_{chunk}_logy.png",
                    f"Jet multiplicity (parquet) — {chunk} — JetGoodMatched",
                    subtitle2,
                    jgm_stats,
                    logy=True,
                )
            except KeyError:
                pass

            # Signal-only extras
            if chunk == "signal_only":
                signal_extras = {"clipping_risk": signal_clipping_risk(pq_files, collection="JetGood", max_index=15)}
                try:
                    matched_stats = parquet_stream_matched_counts(pq_files, max_bin=args.max_bin)
                    matched_payload = matched_stats.to_dict()
                    matched_payload["num_parquet_files"] = int(len(pq_files))
                    matched_payload["frac_any_matched_idx_gt15"] = signal_extras["clipping_risk"]["frac_any_matched_idx_gt_max"]
                    write_json(chunk_dir / "parquet_JetGood_matchedCount_stats.json", matched_payload)
                    matched_stats.write_hist_csv(chunk_dir / "parquet_JetGood_matchedCount_hist.csv")
                    plot_from_hist(
                        chunk_dir / "parquet_JetGood_matchedCount_logy.png",
                        "Matched jet multiplicity (parquet) — signal_only — sum(JetGood.matched)",
                        f"N={matched_payload['n_events']:,}  frac_any_matched_idx>15={matched_payload['frac_any_matched_idx_gt15']:.4%}",
                        matched_stats,
                        logy=True,
                    )
                    signal_extras["matched_count"] = matched_payload
                except Exception:
                    pass

        # H5 MASK multiplicity for this chunk (train+test shards)
        h5_files = []
        if not args.skip_h5:
            h5_files = sorted(h5_base.glob(f"output_sig_QCD_classification_{args.run_tag}_{chunk}_train_*.h5"))
            h5_files += sorted(h5_base.glob(f"output_sig_QCD_classification_{args.run_tag}_{chunk}_test_*.h5"))
        h5_payload = None
        if h5_files:
            h5_stats = h5_stream_mask_counts(h5_files, max_bin=args.max_bin)
            h5_payload = h5_stats.to_dict()
            h5_payload["num_h5_files"] = int(len(h5_files))
            write_json(chunk_dir / "h5_MASK_stats.json", h5_payload)
            h5_stats.write_hist_csv(chunk_dir / "h5_MASK_hist.csv")
            subtitle3 = (
                f"files={len(h5_files)}  N={h5_payload['n_events']:,}  "
                f"min={h5_payload['min']} max={h5_payload['max']}  mean={h5_payload['mean']:.2f}"
            )
            plot_from_hist(
                chunk_dir / f"h5_MASK_{chunk}_logy.png",
                f"Jet multiplicity (H5) — {chunk} — sum(INPUTS/Jet/MASK)",
                subtitle3,
                h5_stats,
                logy=True,
            )
            plot_from_hist(
                chunk_dir / f"h5_MASK_{chunk}_lineary.png",
                f"Jet multiplicity (H5) — {chunk} — sum(INPUTS/Jet/MASK)",
                subtitle3,
                h5_stats,
                logy=False,
            )

        summary["chunks"][chunk] = {
            "parquet_JetGood": jetgood_payload,
            "parquet_JetGoodMatched": jetgoodmatched_payload,
            "signal_only_extras": signal_extras,
            "h5_MASK": h5_payload,
        }

    # Overall parquet JetGood
    if not args.skip_parquet and all_parquet:
        overall_dir = out_dir / "overall"
        ensure_dir(overall_dir)
        overall_stats, _ = parquet_stream_counts(all_parquet, collection="JetGood", max_bin=args.max_bin, pt_sorted_check_events=0)
        overall_payload = overall_stats.to_dict()
        overall_payload["num_parquet_files"] = int(len(all_parquet))
        write_json(overall_dir / "parquet_JetGood_stats.json", overall_payload)
        overall_stats.write_hist_csv(overall_dir / "parquet_JetGood_hist.csv")
        plot_from_hist(
            overall_dir / "parquet_JetGood_overall_logy.png",
            "Jet multiplicity (parquet) — overall — JetGood",
            f"files={len(all_parquet)}  N={overall_payload['n_events']:,}  P(N>16)={overall_payload['frac_gt16']:.4%}",
            overall_stats,
            logy=True,
        )

    write_json(out_dir / "summary.json", summary)

    if args.eos_dir and not args.skip_eos:
        eos_upload_tree(out_dir, args.eos_dir)

    print(f"Done. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()

