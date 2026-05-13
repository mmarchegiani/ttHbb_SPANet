import os
import argparse
import awkward as ak

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True, help="Input folder path")
parser.add_argument("-j", "--workers", type=int, default=8, help="Number of parallel workers")
parser.add_argument("--overwrite-metadata", action="store_true", help="Overwrite metadata if it already exists")
args = parser.parse_args()

error_filename = "error_parquet_metadata.log"


def find_parquet_dirs(root):
    """Return all directories that directly contain at least one .parquet file."""
    parquet_dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.endswith(".parquet") for f in filenames):
            parquet_dirs.append(dirpath)
    return parquet_dirs


def create_parquet_metadata(dataset):
    root = os.path.join(args.input, dataset)
    parquet_dirs = find_parquet_dirs(root)
    if not parquet_dirs:
        print(f"No parquet files found under {dataset}, skipping")
        return
    for dataset_path in parquet_dirs:
        rel = os.path.relpath(dataset_path, args.input)
        if os.path.exists(os.path.join(dataset_path, "_metadata")) and not args.overwrite_metadata:
            print(f"Metadata already exists for {rel}, skipping")
            continue
        print(f"Processing {rel}")
        try:
            ak.to_parquet.dataset(dataset_path)
        except Exception as e:
            print(f"Error processing {rel}: {e}")
            with open(error_filename, "a") as f:
                f.write(f"{rel}\n")


datasets = os.listdir(args.input)
with Pool(args.workers) as pool:
    pool.map(create_parquet_metadata, datasets)
