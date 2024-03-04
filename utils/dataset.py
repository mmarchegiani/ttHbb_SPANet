import os
from collections import defaultdict

import vector
vector.register_numba()
vector.register_awkward()

import numpy as np
import awkward as ak

from omegaconf import OmegaConf

from coffea.util import load
from coffea.processor.accumulator import column_accumulator
from coffea.processor import accumulate

class ParquetDataset:
    def __init__(self, input_file, output_file, cfg, cat="semilep_LHE"):
        self.input_file = input_file
        self.output_file = output_file
        self.cfg = cfg
        self.cat = cat

        self.load_input()
        self.load_config()
        self.check_output()
        self.normalize_genweights()
        self.load_features()
        self.build_arrays()

    def load_input(self):
        '''Load the input file and check if the event category `cat` is present.'''
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file {self.input_file} does not exist.")
        else:
            self.df = load(self.input_file)
            self.columns = self.df["columns"]
            if not self.cat in self.df["cutflow"].keys():
                raise ValueError(f"Event category `{self.cat}` not found in the input file.")
            self.samples = self.columns.keys()
            # Build the array dictionary and the zipped dictionary for each sample
            self.array_dict = {s : defaultdict(dict) for s in self.samples}
            self.zipped_dict = {s : defaultdict(dict) for s in self.samples}
            self.features_dict = {s : defaultdict(dict) for s in self.samples}
            self.features_pad_dict = {s : defaultdict(dict) for s in self.samples}

    def load_config(self):
        '''Load the features, features_pad, awkward_collections and matched_collections_dict
        dictionaries with OmegaConf.'''
        self.cfg = OmegaConf.load(self.cfg)
        self.features = self.cfg["features"]
        self.features_pad = self.cfg["features_pad"]
        self.awkward_collections = self.cfg["awkward_collections"]
        self.matched_collections_dict = self.cfg["matched_collections"]

    def check_output(self):
        # Check if the output file is already existing
        if os.path.exists(self.output_file):
            raise ValueError(f"Output file {self.output_file} already exists.")
        # Check that the format of the output file is `.parquet`
        if not self.output_file.endswith(".parquet"):
            raise ValueError(f"Output file {self.output_file} should have the `.parquet` extension.")
        # Create the output directory if it does not exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def normalize_genweights(self):
        '''Since the array `weight` is filled on the fly with the weight associated with the event, 
        it does not take into account the overall scaling by the sum of genweights (`sum_genweights`).
        In order to correct for this, we have to scale by hand the `weight` array dividing by the sum of genweights.'''
        for sample in self.samples:
            datasets = self.columns[sample].keys()
            for dataset in datasets:
                weight = self.columns[sample][dataset][self.cat]["weight"].value
                weight_new = column_accumulator(weight / self.df["sum_genweights"][dataset])
                self.columns[sample][dataset][self.cat]["weight"] = weight_new

    def load_features(self):
        '''Load the features dictionary with common features and sample-specific features.'''
        
        print("Loading features")
        # Compose the features dictionary with common features and sample-specific features
        for sample in self.samples:
            self.features_dict[sample] = self.features["common"].copy()
            if sample in self.features["by_sample"].keys():
                self.features_dict[sample].update(self.features["by_sample"][sample])

            # Compose the dictionary of features to pad
            self.features_pad_dict[sample] = self.features_pad["common"].copy()
            if sample in self.features_pad["by_sample"].keys():
                self.features_pad_dict[sample].update(self.features_pad["by_sample"][sample])

            # Create a default dictionary of dictionaries to store the arrays
            self.array_dict[sample] = {k : defaultdict(dict) for k in self.features_dict[sample].keys()}

    def build_arrays(self):
        '''Build the Momentum4D arrays for the jets, partons, leptons, met and higgs.'''
        
        print("Samples: ", self.samples)

        for sample in self.samples:
            datasets = self.columns[sample].keys()
            print("Datasets: ", datasets)

            # Accumulate ntuples from different data-taking eras
            cs = accumulate([self.columns[sample][dataset][self.cat] for dataset in datasets])

            # In order to get the numpy array from the column_accumulator, we have to access the `value` attribute.
            for collection, variables in self.features_dict[sample].items():
                for key_feature, key_coffea in variables.items():
                    if (collection == "JetGoodMatched") & (key_coffea == "provenance"):
                        self.array_dict[sample][collection][key_feature] = cs[f"PartonMatched_{key_coffea}"].value
                    else:
                        self.array_dict[sample][collection][key_feature] = cs[f"{collection}_{key_coffea}"].value

                # Add padded features to the array, according to the features dictionary
                if collection in self.features_pad_dict[sample].keys():
                    for key_feature, value in self.features_pad_dict[sample][collection].items():
                        self.array_dict[sample][collection][key_feature] = value * np.ones_like(cs[f"{collection}_pt"].value)

            # The awkward arrays are zipped together to form the Momentum4D arrays.
            # If the collection is not a Momentum4D array, it is zipped as it is,
            # otherwise the Momentum4D arrays are zipped together and unflattened depending on the number of objects in the collection.
            for collection in self.array_dict[sample].keys():
                if collection in self.awkward_collections:
                    self.zipped_dict[sample][collection] = ak.unflatten(ak.zip(self.array_dict[sample][collection], with_name='Momentum4D'), cs[f"{collection}_N"].value)
                else:
                    self.zipped_dict[sample][collection] = ak.zip(self.array_dict[sample][collection], with_name='Momentum4D')
                print(f"Collection: {collection}")
                print("Fields: ", self.zipped_dict[sample][collection].fields)

            for collection in self.zipped_dict[sample].keys():
                # Pad the matched collections with None if there is no matching
                if collection in self.matched_collections_dict.keys():
                    matched_collection = self.matched_collections_dict[collection]
                    masked_arrays = ak.mask(self.zipped_dict[sample][matched_collection], self.zipped_dict[sample][matched_collection].pt==-999, None)
                    self.zipped_dict[sample][matched_collection] = masked_arrays
                    # Add the matched flag and the provenance to the matched jets
                    if collection == "JetGood":
                        is_matched = ~ak.is_none(masked_arrays, axis=1)
                        self.zipped_dict[sample][collection] = ak.with_field(self.zipped_dict[sample][collection], is_matched, "matched")
                        self.zipped_dict[sample][collection] = ak.with_field(self.zipped_dict[sample][collection], ak.fill_none(masked_arrays.prov, -1), "prov")

            # Add the remaining keys to the zipped dictionary
            self.zipped_dict[sample]["event"] = ak.zip({"weight" : cs["weight"].value})

    def save_parquet(self):
        '''Create the parquet files with the zipped dictionary.'''

        for sample in self.samples:
            # The Momentum4D arrays are zipped together to form the final dictionary of arrays.
            print("Zipping the collections into a single dictionary...")
            df_out = ak.zip(self.zipped_dict[sample], depth_limit=1)

            if len(self.samples) > 1:
                output_file = self.output_file.replace(".parquet", f"_{sample}.parquet")
            else:
                output_file = self.output_file
            print(f"Saving the output dataset to file: {os.path.abspath(output_file)}")
            ak.to_parquet(df_out, self.output_file)

class H5Dataset:
    def __init__(self, input, output_file, cfg, cat="semilep_LHE"):
        # Load several input files into a list
        if type(input) == str:
            self.input_files = [input]
        elif type(input) == list:
            self.input_files = input
        self.output_file = output_file
        self.cfg = cfg
        self.cat = cat

        self.load_input()
        self.load_config()
        self.check_output()
        self.load_features()

    def load_input(self):
        '''Load the input file.'''
        
