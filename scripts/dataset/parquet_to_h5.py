import argparse

import vector
vector.register_numba()
vector.register_awkward()

from utils.dataset import H5Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert awkward ntuples in coffea files to parquet files.')
    parser.add_argument('-c', '--cfg', type=str, required=True, help='YAML configuration file with input features')
    parser.add_argument('-i', '--input', type=str, required=True, nargs='+', help='Input parquet file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output h5 file')
    parser.add_argument('--signal', action='store_true', help='Label signal events')
    parser.add_argument('-fm', '--fully_matched', action='store_true', help='Use only fully matched events')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the events in the output file')

    args = parser.parse_args()

    dataset = H5Dataset(
        args.input,
        args.output,
        args.cfg,
        args.fully_matched,
        args.shuffle
    )
    dataset.save_h5_all()
