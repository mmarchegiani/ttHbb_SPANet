import os
import argparse
import h5py
import numpy as np

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

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", nargs="+", help="Input h5 files", required=True)
parser.add_argument("-o", "--output", help="Output file", required=True)
parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the data")
parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
args = parser.parse_args()

# Check if the input files exist
for f in args.input:
    if not os.path.exists(f):
        raise Exception(f"Input file {f} does not exist")
# Create output folder if it does not exist
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

# Check if the output file is in h5 format
if not args.output.endswith(".h5"):
    raise Exception("Output file must be in h5 format")

# Read data from all files
all_data = []
for file_path in args.input:
    with h5py.File(file_path, 'r') as f:
        all_data.append(read_data(f))

# Concatenate data from all files
combined_data = concatenate_data(all_data)
num_entries = len(combined_data["WEIGHTS"]["weight"])  # Assuming all top-level groups have the same number of entries
args.output = args.output.replace(".h5", f"_{num_entries}.h5")

# Shuffle the combined data
if not args.no_shuffle:
    print("Shuffling the data")
    # Generate a list of indices and shuffle it
    shuffled_indices = np.random.permutation(num_entries)
    # Shuffle the combined data using the shuffled indices
    combined_data = shuffle_data(combined_data, shuffled_indices)

# Check if the output file already exists
if not args.overwrite and os.path.exists(args.output):
    raise Exception(f"Output file {args.output} already exists")

# Write the shuffled data to the new h5 file
print("Writing the data to the output file")
with h5py.File(args.output, 'w') as f:
    write_data(f, combined_data)

msg = f"Data from {args.input} has been merged into {args.output}"
if not args.no_shuffle:
    msg = msg.replace("merged", "merged and shuffled")
print(msg)
