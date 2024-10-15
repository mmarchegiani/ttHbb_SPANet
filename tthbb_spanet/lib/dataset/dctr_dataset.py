import os
import yaml
import numpy as np
import awkward as ak

import torch
from sklearn.preprocessing import StandardScaler

from .base import Dataset
from ttbb_dctr.lib.data_preprocessing import get_njet_reweighting, get_njet_reweighting_map
from ttbb_dctr.lib.quantile_transformer import WeightedQuantileTransformer
from ttbb_dctr.lib.data_preprocessing import get_device

class DCTRDataset(Dataset):
    def __init__(self, input_file, cfg=None, shuffle=True, reweigh=False, entrystop=None, has_data=False, label=True):
        super().__init__(input_file, cfg=cfg, shuffle=shuffle, reweigh=reweigh, entrystop=entrystop, has_data=has_data, label=label)
        if (not "spanet_output" in self.df.fields) | ("BJetGood" not in self.df.fields):
            self.create_branches()

    def create_branches(self):
        mask_btag = ak.values_astype(self.df.JetGood.btag_M, bool)
        self.df = ak.with_field(self.df, self.df.JetGood[mask_btag], "BJetGood")
        transformer = WeightedQuantileTransformer(n_quantiles=100000, output_distribution='uniform')
        mask_tthbb = self.df.tthbb == 1
        X = self.df.spanet_output.tthbb[mask_tthbb]
        transformer.fit(X, sample_weight=-self.df.event.weight[mask_tthbb]) # Fit quantile transformer on ttHbb sample only (- in front of weights due to negative weights in DCTR samples)
        transformed_score = transformer.transform(self.df.spanet_output.tthbb)
        self.df["spanet_output"] = ak.with_field(self.df.spanet_output, transformed_score, "tthbb_transformed")
        # Create a copy of the original weights
        self.df["event"] = ak.with_field(self.df.event, self.df.event.weight, "weight_original")

    def compute_njet_weights(self):
        '''Compute the 1D reweighting based on the number of jets.'''
        mask_num = self.df.dctr == 1 # Data - other backgrounds
        mask_den = self.df.dctr == 0 # ttbb
        reweighting_map_dict = {}
        for name, mask in self.masks.items():
            reweighting_map = get_njet_reweighting_map(self.df, mask & mask_num, mask & mask_den)
            self.store_weight(name, get_njet_reweighting(self.df, reweighting_map, mask & mask_den))
            reweighting_map_dict[name] = reweighting_map
        self.reweighting_map = reweighting_map_dict

    def save_reweighting_map(self, output_file):
        '''Save the reweighting map to a json file.'''
        self.check_output(output_file, ext=".yaml")
        print(f"Saving reweighting map to: {output_file}")
        with open(output_file, "w") as f:
            yaml.dump(self.reweighting_map, f)

    def check_output(self, output_file, ext=".parquet"):
        '''Check the output file extension and if it already exists.'''
        # Check the output file extension
        filename, file_extension = os.path.splitext(output_file)
        if not file_extension == ext:
            raise ValueError(f"Output file {output_file} should be in {ext} format.")
        # Check if output file exists
        if os.path.exists(output_file):
            raise ValueError(f"Output file {output_file} already exists.")
        os.makedirs(os.path.abspath(os.path.dirname(output_file)), exist_ok=True)

    def apply_weight(self, df, weight):
        '''Apply the event weights.'''
        df["event"] = ak.with_field(df["event"], df.event.weight * weight, "weight")
        return df

    def save(self, output_file, mask_name=None):
        '''Save the parquet file.'''
        os.makedirs(os.path.abspath(os.path.dirname(output_file)), exist_ok=True)
        for dataset in ["train", "test"]:
            df_to_save = getattr(self, dataset)
            if mask_name is not None:
                mask = self.masks[mask_name][getattr(self, f"{dataset}_mask")]
                if mask_name in self.weights.keys():
                    df_to_save = self.apply_weight(df_to_save, self.weights[mask_name][getattr(self, f"{dataset}_mask")])
                df_to_save = df_to_save[mask]
            output_file_dataset = output_file.replace(".parquet", f"_{mask_name}_{dataset}_{len(df_to_save)}.parquet")
            self.check_output(output_file_dataset, ext=".parquet")
            print(f"Saving {dataset} dataset to: {output_file_dataset}")
            ak.to_parquet(df_to_save, output_file_dataset)

    def save_all(self, output_file):
        for mask_name in self.masks.keys():
            self.save(output_file, mask_name)

    @classmethod
    def from_parquet(self, input_file, shuffle=False, reweigh=False, entrystop=None, has_data=False, label=False):
        '''Load the input file.'''

        return DCTRDataset(input_file, shuffle=shuffle, reweigh=reweigh, entrystop=entrystop, has_data=has_data, label=label)
