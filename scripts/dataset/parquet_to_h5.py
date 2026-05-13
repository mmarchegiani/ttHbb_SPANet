import argparse
import sys
from pathlib import Path


def import_spanet_dataset():
    """Import SPANetDataset with a local fallback for script execution."""
    try:
        from tthbb_spanet.lib.dataset.spanet_dataset import SPANetDataset as dataset_cls
        return dataset_cls
    except ModuleNotFoundError as exc:
        # Only fallback for local package resolution issues, not missing deps.
        if not exc.name or not exc.name.startswith("tthbb_spanet"):
            raise

        repo_root = Path(__file__).resolve().parents[2]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from tthbb_spanet.lib.dataset.spanet_dataset import SPANetDataset as dataset_cls
        return dataset_cls

if __name__ == '__main__':
    SPANetDataset = import_spanet_dataset()

    parser = argparse.ArgumentParser(description='Convert awkward ntuples in coffea files to parquet files.')
    parser.add_argument('-c', '--cfg', type=str, required=True, help='YAML configuration file with input features')
    parser.add_argument('-i', '--input', type=str, required=True, nargs='+', help='Input parquet file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output h5 file')
    parser.add_argument('-fm', '--fully_matched', action='store_true', help='Use only fully matched events')
    parser.add_argument('--no_shuffle', action='store_true', help='If set, do not shuffle the dataset')
    parser.add_argument('--reweigh', action='store_true', help='If set, scale event weights by a factor as specified in the configuration file')
    parser.add_argument('--entrystop', type=int, default=None, required=False, help='Number of events to process')

    args = parser.parse_args()

    dataset = SPANetDataset(
        args.input,
        args.cfg,
        (not args.no_shuffle),
        args.reweigh,
        args.entrystop,
        False,
        args.fully_matched
    )
    dataset.save(args.output)
