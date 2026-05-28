"""Generate event fraction reports (JSON + pie chart) for H5 datasets."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def _build_class_map(enc_dict, sample_to_class):
    """Return {encoding_int -> class_label} collapsing multiple samples sharing the same encoding."""
    result = {}
    for sample_name, enc_val in enc_dict.items():
        class_name = sample_to_class.get(sample_name, sample_name)
        result[enc_val] = class_name
    return result


def generate_report(h5_file, cfg_file, report_dir=None):
    """Generate event counts JSON and pie chart for an H5 dataset file.

    The report directory defaults to a sibling folder next to the H5 file with
    'output' replaced by 'report' in the stem, e.g. output_2024_train_N.h5 ->
    report_2024_train_N/.
    """
    h5_path = Path(h5_file)

    if report_dir is None:
        stem = h5_path.stem
        report_name = "report" + stem[len("output"):] if stem.startswith("output") else f"report_{stem}"
        report_dir = h5_path.parent / report_name

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(cfg_file) if isinstance(cfg_file, (str, Path)) else cfg_file

    if "mapping_encoding" not in cfg:
        print(f"No mapping_encoding in config, skipping report for {h5_path.name}")
        return report_dir

    mapping_encoding = OmegaConf.to_container(cfg["mapping_encoding"])
    mapping_sample = OmegaConf.to_container(cfg.get("mapping_sample", {}))

    all_counts = {}

    with h5py.File(h5_path, "r") as f:
        for label_key, enc_dict in mapping_encoding.items():
            ds_path = f"CLASSIFICATIONS/EVENT/{label_key}"
            if ds_path not in f:
                print(f"Dataset {ds_path} not found in {h5_path.name}, skipping")
                continue

            signal = f[ds_path][:]
            class_map = _build_class_map(enc_dict, mapping_sample)
            total = len(signal)

            print(f"\n[{h5_path.name}] Total events: {total:,}")
            counts = {}
            for enc_val, class_name in sorted(class_map.items()):
                n = int(np.sum(signal == enc_val))
                if n <= 0:
                    breakpoint()
                    raise ValueError(f"There are no events with encoding {enc_val} for label {label_key} in {h5_path.name}, but it is defined in mapping_encoding.")
                counts[class_name] = counts.get(class_name, 0) + n

            for class_name, n in counts.items():
                pct = n / total * 100 if total > 0 else 0.0
                print(f"  {class_name}: {n:,}  ({pct:.2f}%)")

            all_counts[label_key] = {"total": total, "by_class": counts}

            fracs = {k: (v / total * 100) if total > 0 else 0.0 for k, v in counts.items()}
            n_classes = len(fracs)
            colors = list(plt.cm.Set3.colors[:n_classes])

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                list(fracs.values()),
                labels=list(fracs.keys()),
                colors=colors,
                autopct='%1.1f%%',
                startangle=140,
            )
            ax.set_title(f'Distribution of Event Classes ({label_key})\n{h5_path.name}  ({total:,} events)')
            ax.axis('equal')

            pie_path = report_dir / f"event_fractions_{label_key}.png"
            fig.savefig(pie_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"Saved pie chart: {pie_path}")

    json_path = report_dir / "event_counts.json"
    with open(json_path, "w") as jf:
        json.dump(all_counts, jf, indent=2)
    print(f"Saved event counts: {json_path}")

    return report_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate event fraction report (JSON + pie chart) for H5 dataset(s).'
    )
    parser.add_argument('--h5', type=str, required=True, nargs='+', help='Input H5 file(s)')
    parser.add_argument('--cfg', type=str, required=True, help='YAML config file with mapping_encoding')
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for the report (derived from H5 filename if not given)',
    )
    args = parser.parse_args()

    for h5_file in args.h5:
        generate_report(h5_file, args.cfg, args.output_dir)
