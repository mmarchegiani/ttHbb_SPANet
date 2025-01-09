import os
import yaml
import numpy as np
import awkward as ak

from .base import Dataset

def get_njet_reweighting_map(events, mask_num, mask_den):
    reweighting_map_njet = {}
    njet = ak.num(events.JetGood)
    w = events.event.weight
    for nj in range(4,7):
        mask_nj = (njet == nj)
        reweighting_map_njet[nj] = sum(w[mask_num & mask_nj]) / sum(w[mask_den & mask_nj])
    for nj in range(7,21):
        reweighting_map_njet[nj] = sum(w[mask_num & (njet >= 7)]) / sum(w[mask_den & (njet >= 7)])
    return reweighting_map_njet

def get_njet_reweighting(events, reweighting_map_njet, mask=None):
    njet = ak.num(events.JetGood)
    w = events.event.weight
    w_nj = np.ones(len(events))
    if mask is None:
        mask = np.ones(len(events), dtype=bool)
    for nj in range(4,7):
        mask_nj = (njet == nj)
        w_nj = np.where(mask & mask_nj, reweighting_map_njet[nj], w_nj)
    for nj in range(7,21):
        w_nj = np.where(mask & (njet >= 7), reweighting_map_njet[nj], w_nj)
    print("1D reweighting map based on the number of jets:")
    print(reweighting_map_njet)
    return w_nj

class DCTRDataset(Dataset):
    def __init__(self, input_file, cfg=None, shuffle=True, reweigh=False, entrystop=None, has_data=False, label=True):
        super().__init__(input_file, cfg=cfg, shuffle=shuffle, reweigh=reweigh, entrystop=entrystop, has_data=has_data, label=label)
        if (not "spanet_output" in self.df.fields) | ("BJetGood" not in self.df.fields):
            self.create_branches()
        assert all(self.df.event.weight[self.df.data == 1] == self.df.event.weight_original[self.df.data == 1]), "Data weights should be equal to the original weights."

    def create_branches(self):
        # Define b-tagged jets collection
        mask_btag = ak.values_astype(self.df.JetGood.btag_M, bool)
        self.df = ak.with_field(self.df, self.df.JetGood[mask_btag], "BJetGood")
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

    def apply_weight(self, df, weight, mask=None):
        '''Apply the event weights. If a mask is provided, apply the weights only to the events that pass the mask.'''
        if mask is None:
            df["event"] = ak.with_field(df["event"], df.event.weight * weight, "weight")
        else:
            df["event"] = ak.with_field(df["event"], np.where(mask, df.event.weight * weight, df.event.weight), "weight")
        return df

    def save(self, output_file, mask_name=None):
        '''Save the parquet file.'''
        assert all(self.df.event.weight[self.df.data == 1] == self.df.event.weight_original[self.df.data == 1]), "Data weights should be equal to the original weights."
        os.makedirs(os.path.abspath(os.path.dirname(output_file)), exist_ok=True)
        dataframes_to_save = {}
        for dataset in ["train", "test"]:
            df_to_save = getattr(self, dataset)
            if mask_name is not None:
                mask = self.masks[mask_name][getattr(self, f"{dataset}_mask")]
                if mask_name in self.weights.keys():
                    df_to_save = self.apply_weight(df_to_save, self.weights[mask_name][getattr(self, f"{dataset}_mask")])
                df_to_save = df_to_save[mask]
            dataframes_to_save[dataset] = df_to_save
            output_file_dataset = output_file.replace(".parquet", f"_{mask_name}_{dataset}_{len(df_to_save)}.parquet")
            self.check_output(output_file_dataset, ext=".parquet")
            print(f"Saving {dataset} dataset to: {output_file_dataset}")
            ak.to_parquet(df_to_save, output_file_dataset)
        # Save also the full dataset concatenating the train and test datasets
        output_file_full = output_file.replace(".parquet", f"_{mask_name}_full_{len(dataframes_to_save['train'])+len(dataframes_to_save['test'])}.parquet")
        self.check_output(output_file_full, ext=".parquet")
        print(f"Saving full dataset to: {output_file_full}")
        ak.to_parquet(ak.concatenate([dataframes_to_save["train"], dataframes_to_save["test"]]), output_file_full)


    def save_all(self, output_file):
        for mask_name in self.masks.keys():
            self.save(output_file, mask_name)

    @classmethod
    def from_parquet(self, input_file, shuffle=False, reweigh=False, entrystop=None, has_data=False, label=False):
        '''Load the input file.'''

        return DCTRDataset(input_file, shuffle=shuffle, reweigh=reweigh, entrystop=entrystop, has_data=has_data, label=label)
