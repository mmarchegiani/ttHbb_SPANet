import argparse
from pathlib import Path

from tthbb_spanet.lib.dataset.spanet_dataset import SPANetDataset
from tthbb_spanet.scripts.dataset.plot_event_fractions import generate_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert awkward ntuples in coffea files to parquet files.')
    parser.add_argument('-c', '--cfg', type=str, required=True, help='YAML configuration file with input features')
    parser.add_argument('-i', '--input', type=str, required=True, nargs='+', help='Input parquet file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output h5 file')
    parser.add_argument('-fm', '--fully_matched', action='store_true', help='Use only fully matched events')
    parser.add_argument('--no_shuffle', action='store_true', help='If set, do not shuffle the dataset')
    parser.add_argument('--reweigh', action='store_true', help='If set, scale event weights by a factor as specified in the configuration file')
    parser.add_argument('--entrystop', type=int, default=None, required=False, help='Number of events to process')
    parser.add_argument('--batch_size', type=int, default=None, required=False,
                        help='Target number of events per batch when reading parquet (rounded to a whole number of row groups). Default: one row group per batch.')
    parser.add_argument('--no_shuffle_output', action='store_true',
                        help='Skip the final global shuffle of the output h5 files (saves RAM at the cost of leaving the data sorted by input file).')

    args = parser.parse_args()

    dataset = SPANetDataset(
        args.input,
        args.cfg,
        shuffle=(not args.no_shuffle),
        reweigh=args.reweigh,
        entrystop=args.entrystop,
        has_data=False,
        fully_matched=args.fully_matched,
        batch_size=args.batch_size,
        shuffle_output=(not args.no_shuffle_output),
    )
    dataset.save(args.output)

    if dataset.one_hot_encoding:
        output_path = Path(args.output)
        for h5_file in sorted(output_path.parent.glob(f"{output_path.stem}_*.h5")):
            generate_report(h5_file, args.cfg)
