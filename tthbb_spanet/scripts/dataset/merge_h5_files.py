import os
import gc
import argparse
import h5py
import numpy as np


def collect_schema(group, prefix=""):
    """Walk an h5 group and return a list of (path, shape, dtype, chunks) tuples for every dataset."""
    out = []
    for key, item in group.items():
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):
            out.append((path, item.shape, item.dtype, item.chunks))
        elif isinstance(item, h5py.Group):
            out.extend(collect_schema(item, path))
    return out


def ensure_group(h5file, path):
    """Create intermediate groups for a dataset path like '/A/B/name' and return the parent group."""
    parts = [p for p in path.split("/") if p]
    parent = h5file
    for p in parts[:-1]:
        parent = parent.require_group(p)
    return parent, parts[-1]


def fill_buffer_from_inputs(input_files, path, row_counts, file_offsets, buf, read_chunk):
    """Read the same dataset path from each input file and place it into buf at the right offset."""
    for i, f_path in enumerate(input_files):
        n = row_counts[i]
        offset = file_offsets[i]
        with h5py.File(f_path, "r") as in_f:
            ds = in_f[path]
            if ds.shape[0] != n:
                raise Exception(
                    f"Dataset {path} in {f_path} has {ds.shape[0]} rows, "
                    f"expected {n} (mismatch with reference dataset in same file)."
                )
            # Stream the read in chunks to keep peak memory bounded even for huge inputs.
            for s in range(0, n, read_chunk):
                e = min(s + read_chunk, n)
                ds.read_direct(buf, np.s_[s:e], np.s_[offset + s:offset + e])


def write_dataset(out_f, path, buf, perm, chunks, write_chunk):
    """Create a dataset in out_f at `path` and write buf, applying `perm` if given."""
    parent, name = ensure_group(out_f, path)
    out_ds = parent.create_dataset(name, shape=buf.shape, dtype=buf.dtype, chunks=chunks)
    if perm is None:
        # Sequential write in chunks (avoids h5py shipping the full array in one call).
        total = buf.shape[0]
        for s in range(0, total, write_chunk):
            e = min(s + write_chunk, total)
            out_ds[s:e] = buf[s:e]
    else:
        total = buf.shape[0]
        for s in range(0, total, write_chunk):
            e = min(s + write_chunk, total)
            # out[s:e] = buf[perm[s:e]]  -> sequential output write, gather-read from buf.
            out_ds[s:e] = buf[perm[s:e]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs="+", help="Input h5 files", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the data")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the shuffle permutation")
    parser.add_argument("--read-chunk", type=int, default=1_000_000,
                        help="Rows to read per input slice (per dataset). Default: 1,000,000")
    parser.add_argument("--write-chunk", type=int, default=1_000_000,
                        help="Rows to write per output slice (per dataset). Default: 1,000,000")
    return parser.parse_args()


def main():
    args = parse_args()

    for f in args.input:
        if not os.path.exists(f):
            raise Exception(f"Input file {f} does not exist")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if not args.output.endswith(".h5"):
        raise Exception("Output file must be in h5 format")

    # Pass 1: build the reference schema from the first file and collect row counts.
    print("Scanning input files...")
    with h5py.File(args.input[0], "r") as f:
        reference_schema = collect_schema(f)
    if not reference_schema:
        raise Exception(f"No datasets found in {args.input[0]}")
    ref_paths = {p: (s[1:], np.dtype(d)) for p, s, d, _ in reference_schema}

    # The leading dim of the first dataset in the reference is treated as the event count.
    row_counts = []
    for f_path in args.input:
        with h5py.File(f_path, "r") as f:
            schema = collect_schema(f)
            these = {p: (s[1:], np.dtype(d)) for p, s, d, _ in schema}
            if these.keys() != ref_paths.keys():
                missing = ref_paths.keys() - these.keys()
                extra = these.keys() - ref_paths.keys()
                raise Exception(
                    f"Schema mismatch in {f_path}: missing={sorted(missing)}, extra={sorted(extra)}"
                )
            for p, (tail, dt) in ref_paths.items():
                if these[p] != (tail, dt):
                    raise Exception(
                        f"Schema mismatch for {p} in {f_path}: "
                        f"got tail={these[p][0]} dtype={these[p][1]}, "
                        f"expected tail={tail} dtype={dt}"
                    )
            n = schema[0][1][0]
            for p, s, _, _ in schema:
                if s[0] != n:
                    raise Exception(
                        f"In {f_path}, dataset {p} has leading dim {s[0]}, "
                        f"expected {n} (all datasets must be event-aligned)."
                    )
            row_counts.append(n)
            print(f"  {f_path}: {n} events")

    total = int(sum(row_counts))
    file_offsets = np.cumsum([0] + row_counts[:-1]).astype(np.int64)
    print(f"Total events: {total}")

    # Update the output path with the total count, matching the original convention.
    args.output = args.output.replace(".h5", f"_{total}.h5")
    if not args.overwrite and os.path.exists(args.output):
        raise Exception(f"Output file {args.output} already exists")

    # Build the permutation once (int64; ~8 bytes/event -> ~750 MB for 93M events).
    if args.no_shuffle:
        perm = None
    else:
        print("Generating shuffle permutation")
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(total)

    # Pass 2: stream one dataset at a time through a merged buffer.
    print(f"Writing to {args.output}")
    with h5py.File(args.output, "w") as out_f:
        for path, ref_shape, dtype, ref_chunks in reference_schema:
            tail = tuple(ref_shape[1:])
            shape = (total,) + tail
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            print(f"  {path}: shape={shape} dtype={dtype} ({nbytes / 1e9:.2f} GB)")
            buf = np.empty(shape, dtype=dtype)
            try:
                fill_buffer_from_inputs(
                    args.input, path, row_counts, file_offsets, buf, args.read_chunk,
                )
                write_dataset(out_f, path, buf, perm, ref_chunks, args.write_chunk)
            finally:
                del buf
                gc.collect()

    action = "merged" if args.no_shuffle else "merged and shuffled"
    print(f"Data from {len(args.input)} files has been {action} into {args.output}")


if __name__ == "__main__":
    main()
