"""Plot 1D distributions of jet input features, signal vs background, split by jet index.

Usage:
    python plot_input_features.py [--max-events N] [--max-jets J] [--output-dir DIR]

Reads parquet files from PARQUET_DIR, sampling up to --max-events per class.
Jets are pT-ordered by the upstream coffea processing.
"""

import argparse
import gc
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

PARQUET_DIR = "/home/export/sdurgut/scratch/coffea_to_parquet_output"

FEATURES = ["btag", "btag_L", "btag_M", "btag_T", "btag_QvG"]

FEATURE_CONFIG = {
    "pt":       {"xlabel": r"Jet $p_{\mathrm{T}}$ [GeV]", "log_x": True,  "bins": np.geomspace(20, 2000, 50)},
    "eta":      {"xlabel": r"Jet $\eta$",                  "log_x": False, "bins": np.linspace(-3, 3, 50)},
    "phi":      {"xlabel": r"Jet $\phi$",                  "log_x": False, "bins": np.linspace(-3.2, 3.2, 50)},
    "btag":     {"xlabel": "RobustPartT AK4 b-tag score",      "log_x": False, "bins": np.linspace(0, 1, 50)},
    "btag_L":   {"xlabel": "b-tag Loose WP",               "log_x": False, "bins": np.array([-0.5, 0.5, 1.5])},
    "btag_M":   {"xlabel": "b-tag Medium WP",              "log_x": False, "bins": np.array([-0.5, 0.5, 1.5])},
    "btag_T":   {"xlabel": "b-tag Tight WP",               "log_x": False, "bins": np.array([-0.5, 0.5, 1.5])},
    "btag_CvL": {"xlabel": "CvL score",                    "log_x": False, "bins": np.linspace(0, 1, 50)},
    "btag_CvB": {"xlabel": "CvB score",                    "log_x": False, "bins": np.linspace(0, 1, 50)},
    "btag_QvG": {"xlabel": "QvG score",                    "log_x": False, "bins": np.linspace(0, 1, 50)},
}

SIGNAL_COLOR = "#e41a1c"
BKG_COLOR = "#377eb8"


def collect_files(parquet_dir):
    signal_files, bkg_files = [], []
    for fname in sorted(os.listdir(parquet_dir)):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(parquet_dir, fname)
        if "TTH_Hto2B" in fname:
            signal_files.append(path)
        elif "QCD" in fname:
            bkg_files.append(path)
    return signal_files, bkg_files


def build_histograms(file_list, max_events, max_jets):
    """Stream parquet files one row group at a time, using flattened arrow arrays."""
    bins_dict = {feat: FEATURE_CONFIG[feat]["bins"] for feat in FEATURES}
    histograms = {
        feat: np.zeros((max_jets, len(bins_dict[feat]) - 1), dtype=np.int64)
        for feat in FEATURES
    }
    n_collected = 0
    unlimited = max_events <= 0

    for fpath in file_list:
        if not unlimited and n_collected >= max_events:
            break
        pf = pq.ParquetFile(fpath)

        for rg_idx in range(pf.metadata.num_row_groups):
            if not unlimited and n_collected >= max_events:
                break

            table = pf.read_row_group(rg_idx, columns=["JetGood"])
            jet_list = table.column("JetGood").combine_chunks()

            rg_events = len(jet_list)
            if not unlimited:
                rg_events = min(rg_events, max_events - n_collected)

            offsets = jet_list.offsets.to_numpy()[:rg_events + 1]
            flat_structs = jet_list.flatten()
            flat_end = int(offsets[-1])

            jet_idx = np.empty(flat_end, dtype=np.int32)
            for i in range(rg_events):
                jet_idx[offsets[i]:offsets[i + 1]] = np.arange(offsets[i + 1] - offsets[i])

            for feat in FEATURES:
                vals = flat_structs.field(feat).to_numpy()[:flat_end].astype(np.float64)
                bins = bins_dict[feat]
                for j in range(max_jets):
                    mask = jet_idx == j
                    if mask.any():
                        h, _ = np.histogram(vals[mask], bins=bins)
                        histograms[feat][j] += h

            n_collected += rg_events
            del table, jet_list, flat_structs, offsets, jet_idx
            gc.collect()

            print(f"    {n_collected:,} events ({os.path.basename(fpath)} rg {rg_idx})")

    print(f"  Total: {n_collected:,} events from {len(file_list)} files")
    return histograms


def plot_feature_per_jet(sig_hists, bkg_hists, max_jets, output_dir):
    def normalize_to_unit_area(counts, bins):
        """Return a histogram scaled so integral over x is 1.

        For variable-width bins, this returns a probability density:
          sum_i (density_i * bin_width_i) = 1
        """
        counts = np.asarray(counts, dtype=np.float64)
        widths = np.diff(bins).astype(np.float64)
        area = np.sum(counts * widths)
        if area <= 0:
            return np.zeros_like(counts, dtype=np.float64)
        return counts / area

    for feat in FEATURES:
        cfg = FEATURE_CONFIG[feat]
        bins = cfg["bins"]

        ncols = min(max_jets, 4)
        nrows = (max_jets + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)

        for j in range(max_jets):
            ax = axes[j // ncols][j % ncols]

            bkg = normalize_to_unit_area(bkg_hists[feat][j], bins)
            sig = normalize_to_unit_area(sig_hists[feat][j], bins)

            hep.histplot(bkg, bins=bins, ax=ax, label="QCD multijet",
                         histtype="fill", color=BKG_COLOR, alpha=0.3, edgecolor=BKG_COLOR, linewidth=2)
            hep.histplot(sig, bins=bins, ax=ax, label=r"$t\bar{t}H(b\bar{b})$",
                         histtype="step", color=SIGNAL_COLOR, linewidth=2)

            ax.set_xlabel(cfg["xlabel"])
            ax.set_ylabel("Normalized density (area = 1)")
            if cfg["log_x"]:
                ax.set_xscale("log")

            ax.set_ylim(bottom=0)
            ax.legend(loc="upper right", fontsize=14)

            hep.cms.label("Work in Progress", data=False, ax=ax, fontsize=15, loc=0)
            ax.text(0.04, 0.78, f"Jet {j}", transform=ax.transAxes, fontsize=16, fontweight="bold")

        for j in range(max_jets, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.tight_layout()

        out_path = os.path.join(output_dir, f"input_feature_{feat}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet-dir", default=PARQUET_DIR)
    parser.add_argument("--max-events", type=int, default=0,
                        help="Max events to sample per class (default: 0 = all)")
    parser.add_argument("--max-jets", type=int, default=8,
                        help="Number of leading jets to plot (default: 8)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for plots (default: <script_dir>/input_feature_plots/<timestamp>)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing plot files (default: skip existing files)")
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(os.path.dirname(__file__), "input_feature_plots", ts)
    os.makedirs(args.output_dir, exist_ok=True)

    signal_files, bkg_files = collect_files(args.parquet_dir)
    print(f"Found {len(signal_files)} signal files, {len(bkg_files)} background files")

    print("Reading signal...")
    sig_hists = build_histograms(signal_files, args.max_events, args.max_jets)

    print("Reading background...")
    bkg_hists = build_histograms(bkg_files, args.max_events, args.max_jets)

    print("Plotting...")
    if not args.overwrite:
        # If the output directory already contains any of our expected files, avoid clobbering.
        existing = [feat for feat in FEATURES if os.path.exists(os.path.join(args.output_dir, f"input_feature_{feat}.png"))]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing plots in '{args.output_dir}'. "
                f"Found {len(existing)} existing plot(s). Use --overwrite or choose a new --output-dir."
            )

    plot_feature_per_jet(sig_hists, bkg_hists, args.max_jets, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
