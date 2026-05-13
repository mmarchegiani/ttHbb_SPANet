import os
import argparse
import h5py
import numpy as np
import time
from datetime import datetime

# Function to recursively read datasets from groups and subgroups
def read_data(group):
    data = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            data[key] = item[:]
        elif isinstance(item, h5py.Group):
            data[key] = read_data(item)
    return data

# Function to recursively concatenate datasets from multiple files
def concatenate_data(data_list):
    if isinstance(data_list[0], np.ndarray):
        return np.concatenate(data_list, axis=0)
    concatenated = {}
    for key in data_list[0].keys():
        concatenated[key] = concatenate_data([data[key] for data in data_list])
    return concatenated

# Function to recursively shuffle datasets
def shuffle_data(data, indices):
    if isinstance(data, np.ndarray):
        return data[indices]
    shuffled = {}
    for key in data.keys():
        shuffled[key] = shuffle_data(data[key], indices)
    return shuffled

# Function to recursively write datasets to groups and subgroups
def write_data(group, data):
    for key, item in data.items():
        if isinstance(item, np.ndarray):
            group.create_dataset(key, data=item)
        else:
            subgroup = group.create_group(key)
            write_data(subgroup, item)


def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", nargs="+", help="Input h5 files", required=True)
parser.add_argument("-o", "--output", help="Output file", required=True)
parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the data")
parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
args = parser.parse_args()
job_t0 = time.time()
log(f"Starting merge with {len(args.input)} input files")

# Check if the input files exist
for f in args.input:
    if not os.path.exists(f):
        raise Exception(f"Input file {f} does not exist")
log("All input files exist")
# Create output folder if it does not exist
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

# Check if the output file is in h5 format
if not args.output.endswith(".h5"):
    raise Exception("Output file must be in h5 format")
log(f"Output base path: {args.output}")

# Read data from all files
all_data = []
for i, file_path in enumerate(args.input, start=1):
    t0 = time.time()
    log(f"[{i}/{len(args.input)}] Reading file: {file_path}")
    with h5py.File(file_path, 'r') as f:
        file_data = read_data(f)
        all_data.append(file_data)
    try:
        n_file = len(file_data["WEIGHTS"]["weight"])
        log(f"[{i}/{len(args.input)}] Done reading ({n_file} entries, {time.time() - t0:.1f}s)")
    except Exception:
        log(f"[{i}/{len(args.input)}] Done reading ({time.time() - t0:.1f}s)")

# Concatenate data from all files
t_concat = time.time()
log("Concatenating input datasets...")
combined_data = concatenate_data(all_data)
num_entries = len(combined_data["WEIGHTS"]["weight"])  # Assuming all top-level groups have the same number of entries
args.output = args.output.replace(".h5", f"_{num_entries}.h5")
log(f"Concatenation complete ({num_entries} entries, {time.time() - t_concat:.1f}s)")

# Shuffle the combined data
if not args.no_shuffle:
    t_shuffle = time.time()
    log("Shuffling the merged dataset...")
    # Generate a list of indices and shuffle it
    shuffled_indices = np.random.permutation(num_entries)
    # Shuffle the combined data using the shuffled indices
    combined_data = shuffle_data(combined_data, shuffled_indices)
    log(f"Shuffling complete ({time.time() - t_shuffle:.1f}s)")
else:
    log("Skipping shuffle (--no-shuffle set)")

# Check if the output file already exists
if not args.overwrite and os.path.exists(args.output):
    raise Exception(f"Output file {args.output} already exists")

# Write the shuffled data to the new h5 file
t_write = time.time()
log(f"Writing merged output: {args.output}")
with h5py.File(args.output, 'w') as f:
    write_data(f, combined_data)
log(f"Write complete ({time.time() - t_write:.1f}s)")

msg = f"Data from {args.input} has been merged into {args.output}"
if not args.no_shuffle:
    msg = msg.replace("merged", "merged and shuffled")
log(msg)
log(f"Total elapsed: {time.time() - job_t0:.1f}s")
