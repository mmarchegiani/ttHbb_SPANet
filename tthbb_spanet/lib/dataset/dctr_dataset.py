import os
import awkward as ak

from .base import Dataset

class DCTRDataset(Dataset):
    def check_output(self, output_file):
        '''Check the output file extension and if it already exists.'''
        # Check the output file extension
        filename, file_extension = os.path.splitext(output_file)
        if not file_extension == ".parquet":
            raise ValueError(f"Output file {output_file} should be in .parquet format.")
        # Check if output file exists
        if os.path.exists(output_file):
            raise ValueError(f"Output file {output_file} already exists.")
        os.makedirs(os.path.abspath(os.path.dirname(output_file)), exist_ok=True)

    def save(self, output_file, mask_name=None):
        '''Save the parquet file.'''
        os.makedirs(os.path.abspath(os.path.dirname(output_file)), exist_ok=True)
        for dataset in ["train", "test"]:
            df_to_save = getattr(self, dataset)
            if mask_name is not None:
                mask = self.masks[mask_name][getattr(self, f"{dataset}_mask")]
                df_to_save = df_to_save[mask]
            output_file_dataset = output_file.replace(".parquet", f"_{mask_name}_{dataset}_{len(df_to_save)}.parquet")
            self.check_output(output_file_dataset)
            print(f"Saving {dataset} dataset to: {output_file_dataset}")
            ak.to_parquet(df_to_save, output_file_dataset)

    def save_all(self, output_file):
        for mask_name in self.masks.keys():
            self.save(output_file, mask_name)

    @classmethod
    def from_parquet(self, input_file, shuffle=False, reweigh=False, entrystop=None, has_data=False, label=False):
        '''Load the input file.'''

        return DCTRDataset(input_file, shuffle=shuffle, reweigh=reweigh, entrystop=entrystop, has_data=has_data, label=label)
