#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def read_data(group):
    data = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            data[key] = item[:]
        elif isinstance(item, h5py.Group):
            data[key] = read_data(item)
    return data


def _read_single_file(file_path: str):
    with h5py.File(file_path, "r") as f:
        return read_data(f)


def concatenate_data(data_list):
    if isinstance(data_list[0], np.ndarray):
        return np.concatenate(data_list, axis=0)
    concatenated = {}
    for key in data_list[0].keys():
        concatenated[key] = concatenate_data([data[key] for data in data_list])
    return concatenated


def shuffle_data(data, indices):
    if isinstance(data, np.ndarray):
        return data[indices]
    shuffled = {}
    for key in data.keys():
        shuffled[key] = shuffle_data(data[key], indices)
    return shuffled


def write_data(group, data):
    for key, item in data.items():
        if isinstance(item, np.ndarray):
            group.create_dataset(key, data=item)
        else:
            subgroup = group.create_group(key)
            write_data(subgroup, item)


def default_workers():
    # Prefer Slurm allocation if present; fall back to local core count.
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(k)
        if v:
            try:
                n = int(v)
                if n > 0:
                    return n
            except ValueError:
                pass
    return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs="+", help="Input h5 files", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the data")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Number of parallel file-read workers (default: Slurm CPUs, else local cores)",
    )
    args = parser.parse_args()

    for f in args.input:
        if not os.path.exists(f):
            raise Exception(f"Input file {f} does not exist")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if not args.output.endswith(".h5"):
        raise Exception("Output file must be in h5 format")

    if args.workers < 1:
        raise Exception("--workers must be >= 1")

    # Read data from all files (parallelized per-file).
    if args.workers == 1 or len(args.input) == 1:
        all_data = [_read_single_file(p) for p in args.input]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            all_data = list(ex.map(_read_single_file, args.input))

    combined_data = concatenate_data(all_data)
    num_entries = len(combined_data["WEIGHTS"]["weight"])
    output_path = args.output.replace(".h5", f"_{num_entries}.h5")

    if not args.no_shuffle:
        print("Shuffling the data")
        shuffled_indices = np.random.permutation(num_entries)
        combined_data = shuffle_data(combined_data, shuffled_indices)

    if not args.overwrite and os.path.exists(output_path):
        raise Exception(f"Output file {output_path} already exists")

    print("Writing the data to the output file")
    with h5py.File(output_path, "w") as f:
        write_data(f, combined_data)

    msg = f"Data from {args.input} has been merged into {output_path}"
    if not args.no_shuffle:
        msg = msg.replace("merged", "merged and shuffled")
    print(msg)


if __name__ == "__main__":
    main()

