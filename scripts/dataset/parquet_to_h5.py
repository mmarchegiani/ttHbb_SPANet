import argparse

import vector
vector.register_numba()
vector.register_awkward()

from utils.dataset import H5Dataset

# Read arguments from command line: input file and output directory. Description: script to convert ntuples from coffea file to parquet file.
parser = argparse.ArgumentParser(description='Convert awkward ntuples in coffea files to parquet files.')
parser.add_argument('-c', '--cfg', type=str, required=True, help='YAML configuration file with input features')
parser.add_argument('-i', '--input', type=str, required=True, nargs='+', help='Input parquet file')
parser.add_argument('-o', '--output', type=str, required=True, help='Output h5 file')
parser.add_argument('-fm', '--fully_matched', action='store_true', required=False, help='Use only fully matched events')

args = parser.parse_args()

dataset = H5Dataset(
    args.input,
    args.output,
    args.cfg,
    True
)
dataset.save_h5_all()
