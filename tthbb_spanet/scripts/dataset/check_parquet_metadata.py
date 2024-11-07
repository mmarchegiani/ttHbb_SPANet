import os
import argparse
import awkward as ak

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True, help="Input folder path")
parser.add_argument("--log", type=str, default="error_parquet_metadata.log", required=False, help="Error log")
parser.add_argument("-j", "--workers", type=int, default=8, help="Number of parallel workers")
args = parser.parse_args()

def check_parquet_metadata(dataset):
    if dataset.startswith("TTToSemiLeptonic") or dataset.startswith("TTbbSemiLeptonic"):
        subfolders = os.listdir(os.path.join(args.input, dataset))
        for subfolder in subfolders:
            print(f"Processing {dataset}/{subfolder}")
            dataset_path = os.path.join(args.input, dataset, subfolder, "semilep")
            if os.path.exists(os.path.join(dataset_path, "_metadata")):
                try:
                    ak.from_parquet(dataset_path)
                    print(f"Metadata OK: {dataset}/{subfolder}")
                    return None
                except:
                    print(f"Metadata corrupted: {dataset}/{subfolder}")
                    return f"{dataset}/{subfolder}"
    else:
        print(f"Processing {dataset}")
        dataset_path = os.path.join(args.input, dataset, "semilep")
        if os.path.exists(os.path.join(dataset_path, "_metadata")):
            print(f"Metadata OK: {dataset}")
            try:
                ak.from_parquet(dataset_path)
                return None
            except:
                print(f"Metadata corrupted: {dataset}")
                return f"{dataset}"

datasets =os.listdir(args.input)
# Parallelize the code: one process per dataset
with Pool(args.workers) as pool:
    corrupted_datasets = pool.map(check_parquet_metadata, datasets)
    with open(args.log, "w") as f:
        for dataset in corrupted_datasets:
            if dataset is not None:
                f.write(f"{dataset}\n")
