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

def create_parquet_metadata(dataset):
    if dataset.startswith("TTToSemiLeptonic") or dataset.startswith("TTbbSemiLeptonic"):
        subfolders = os.listdir(os.path.join(args.input, dataset))
        for subfolder in subfolders:
            print(f"Processing {dataset}/{subfolder}")
            dataset_path = os.path.join(args.input, dataset, subfolder, "semilep")
            if os.path.exists(os.path.join(dataset_path, "_metadata")) and not args.overwrite_metadata:
                print(f"Metadata already exists for {dataset}/{subfolder}")
                return
            try:
                ak.to_parquet.dataset(dataset_path)
            except:
                print(f"Error processing {dataset}/{subfolder}")
                with open(error_filename, "a") as f:
                    f.write(f"{dataset}/{subfolder}\n")
    else:
        print(f"Processing {dataset}")
        dataset_path = os.path.join(args.input, dataset, "semilep")
        if os.path.exists(os.path.join(dataset_path, "_metadata")) and not args.overwrite_metadata:
            print(f"Metadata already exists for {dataset}")
            return
        try:
            ak.to_parquet.dataset(dataset_path)
        except:
            print(f"Error processing {dataset}")
            with open(error_filename, "a") as f:
                f.write(f"{dataset}\n")

datasets =os.listdir(args.input)
# Parallelize the code: one process per dataset
with Pool(args.workers) as pool:
    pool.map(create_parquet_metadata, datasets)
