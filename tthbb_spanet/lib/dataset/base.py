import os
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod

import vector
vector.register_awkward()
try:
    import numba  # noqa: F401
    vector.register_numba()
except Exception:
    # Numba is optional for dataset conversion and can be broken in mixed envs.
    pass

import numpy as np
import awkward as ak
import h5py
import pyarrow.parquet as pq

import omegaconf
from omegaconf import OmegaConf

class Dataset:
    def __init__(self, input_file, cfg=None, shuffle=True, reweigh=False, entrystop=None, has_data=False, label=True, lazy=False):
        # Load several input files into a list
        if type(input_file) == str:
            self.input_files = [input_file]
        elif type(input_file) == list:
            self.input_files = input_file
        else:
            raise ValueError(f"Input file {input_file} should be a string or a list of strings.")
        self.cfg = cfg
        self.shuffle = shuffle
        self.reweigh = reweigh
        self.entrystop = entrystop
        self.has_data = has_data
        self.label = label
        self.masks = {}
        self.weights = {}

        self.sample_dict = defaultdict(dict)
        self.load_config()
        if not lazy:
            self.df = self.load_input()

    def get_sample_name(self, input_file):
        '''Get the sample name from the input file name.'''
        sample_list = []
        for sample in self.mapping_sample.keys():
            if sample in input_file:
                sample_list.append(sample)
        if len(sample_list) == 0:
            raise ValueError(f"Sample name not found in the input file name: {input_file}.\nAvailable samples: {self.mapping_sample.keys()}")
        elif len(sample_list) > 1:
            raise ValueError(f"""Multiple sample names found in the input file name: {input_file}.
                                A single sample name should be specified in the file name.\n
                                Available samples: {self.mapping_sample.keys()}""")
        return sample_list[0]

    def get_year(self, input_file):
        '''Get the sample data-taking year from the input file name.'''
        year_list = []
        years = [
            "2016_PreVFP",
            "2016_PostVFP",
            "2017",
            "2018",
            "2022_preEE",
            "2022_postEE",
            "2023_preBPix",
            "2023_postBPix",
            "2024",
            "2025"
        ]
        for year in years:
            if year in input_file:
                year_list.append(year)
        if len(year_list) == 0:
            raise ValueError(f"Year not found in the input file name: {input_file}.\nAvailable years: {', '.join(years)}")
        elif len(year_list) > 1:
            raise ValueError(f"""Multiple years found in the input file name: {input_file}.
                                A single year should be specified in the file name.\n
                                Available years: {', '.join(years)}""")
        return [year_list[0]]

    def build_labels(self, df, sample):
        '''Build labels for the classification nodes.'''
        for s, label in self.mapping_sample.items():
            if label not in df.fields:
                df[label] = ak.values_astype(np.zeros(len(df), dtype=int), int)
            if s == sample:
                new_labels = ak.values_astype(np.ones(len(df), dtype=int), int)
            else:
                new_labels = ak.values_astype(np.zeros(len(df), dtype=int), int)
            # Take the OR of the labels to avoid overwriting the labels when different samples have the same label
            df[label] = df[label] | new_labels

        # Define one hot encoded label for multiclassifier
        if self.one_hot_encoding:
            for label, encoding_dict in self.mapping_encoding.items():
                df[label] = ak.values_astype(np.zeros(len(df), dtype=int), int)
                df[label] = ak.values_astype(np.ones(len(df), dtype=int) * encoding_dict[sample], int)
        return df

    def scale_weights(self, df, sample):
        '''Scale the event weights by a factor as specified in the configuration file.'''
        if sample not in self.weights_scale.keys():
            raise ValueError(f"Sample {sample} not found in the weights_scale dictionary.")
        for s, factor in self.weights_scale.items():
            if s == sample:
                # Overwrite the event weights with the scaled weights
                df["event"] = ak.with_field(df.event, factor * df.event.weight, "weight")
        return df

    def _validate_input_file(self, input_file):
        if not os.path.exists(input_file):
            raise ValueError(f"Input file {input_file} does not exist.")
        if not input_file.endswith(".parquet"):
            raise ValueError(f"Input file {input_file} should have the `.parquet` extension.")

    def _apply_df_processing(self, df, input_file):
        '''Apply labels, weights and metadata to a (possibly partial) awkward array.'''
        if self.has_data and "event" not in df.fields:
            df["event"] = ak.zip({"weight": np.ones(len(df), dtype=np.float64)})
        if self.reweigh:
            df = self.scale_weights(df, self.get_sample_name(input_file))
        if self.label:
            df = self.build_labels(df, self.get_sample_name(input_file))
            df["metadata"] = ak.zip({"year": len(df)*[self.get_year(input_file)]})
            year = df.metadata.year
            df["metadata"] = ak.with_field(df.metadata, year[:,0], "year")
        return df

    def _process_file(self, input_file, max_events=None):
        '''Load and process a single parquet file, applying labels and weights.'''
        self._validate_input_file(input_file)
        print("Reading file: ", input_file)
        df = ak.from_parquet(input_file)
        df = self._apply_df_processing(df, input_file)
        if max_events is not None and len(df) > max_events:
            df = df[:max_events]
        return df

    def _iter_file_batches(self, input_file, max_events=None, batch_size=None):
        '''Yield processed batches of a parquet file one row-group group at a time.

        If batch_size is None, one row group is read per batch. Otherwise, enough
        consecutive row groups are concatenated to approximate batch_size events
        (using the file's average row-group size).
        '''
        self._validate_input_file(input_file)

        pf = pq.ParquetFile(input_file)
        num_rg = pf.num_row_groups
        total_rows = pf.metadata.num_rows

        if num_rg == 0:
            return

        if batch_size is None:
            rg_per_batch = 1
        else:
            avg_rg_rows = total_rows / num_rg
            rg_per_batch = max(1, int(round(batch_size / avg_rg_rows)))

        print(f"Reading file: {input_file} "
              f"({num_rg} row groups, {total_rows} events, {rg_per_batch} row-group(s)/batch)")

        yielded = 0
        for start in range(0, num_rg, rg_per_batch):
            if max_events is not None and yielded >= max_events:
                break
            end = min(start + rg_per_batch, num_rg)
            df = ak.from_parquet(input_file, row_groups=list(range(start, end)))
            df = self._apply_df_processing(df, input_file)
            if max_events is not None and yielded + len(df) > max_events:
                df = df[:max_events - yielded]
            yielded += len(df)
            yield df

    def load_input(self):
        '''Load and concatenate all input parquet files into memory.'''
        print(f"Reading {len(self.input_files)} parquet files: ", self.input_files)
        dfs = []
        remaining_entries = self.entrystop
        for input_file in self.input_files:
            if remaining_entries is not None and remaining_entries <= 0:
                break
            df = self._process_file(input_file, remaining_entries)
            if remaining_entries is not None:
                remaining_entries -= len(df)
            dfs.append(df)

        if len(dfs) == 0:
            raise ValueError("No events were loaded from input parquet files.")
        df_concat = ak.concatenate(dfs) if len(dfs) > 1 else dfs[0]
        del dfs
        if self.shuffle:
            df_concat = df_concat[np.random.permutation(len(df_concat))]
        if self.entrystop:
            df_concat = df_concat[:self.entrystop]
        return df_concat

    def load_defaults(self):
        '''Load default configuration parameters.'''
        self.mapping_sample = None
        self.one_hot_encoding = False
        self.test_size = 0.2

    def load_config(self):
        '''Load the config file with OmegaConf and read the input features.'''
        if self.cfg == None:
            print("No configuration file provided. Loading default configuration parameters.")
            self.load_defaults()
            return
        if type(self.cfg) == str:
            self.cfg = OmegaConf.load(self.cfg)
        elif type(self.cfg) == dict:
            self.cfg = OmegaConf.create(self.cfg)
        print("Reading configuration file: ")
        print(OmegaConf.to_yaml(self.cfg))
        self.mapping_sample = self.cfg["mapping_sample"]
        self.one_hot_encoding = True if "mapping_encoding" in self.cfg else False
        if self.one_hot_encoding:
            self.mapping_encoding = self.cfg["mapping_encoding"]
            # Ensure that self.mapping_encoding is a dictionary of dictionaries where the keys of the dictionaries are the name of the labels to build
            assert all([type(v) == omegaconf.dictconfig.DictConfig for v in self.mapping_encoding.values()]), "mapping_encoding should be a dictionary of dictionaries."
        if "weights_scale" in self.cfg:
            self.weights_scale = self.cfg["weights_scale"]
        self.test_size = self.cfg.get("test_size", 0.2)

    def store_mask(self, name, mask):
        '''Store the mask in the masks dictionary.'''
        assert len(mask) == len(self.df), f"Mask length {len(mask)} does not match the dataframe length {len(self.df)}."
        self.masks[name] = mask

    def store_masks(self, masks):
        '''Store the masks in the masks dictionary.'''
        assert ((type(masks) == dict) | (type(masks) == omegaconf.dictconfig.DictConfig)), f"Masks should be a dictionary."
        for name, mask in masks.items():
            self.store_mask(name, mask)

    def store_weight(self, name, weight):
        '''Store the weight in the weights dictionary.'''
        assert len(weight) == len(self.df), f"Weight length {len(weight)} does not match the dataframe length {len(self.df)}."
        self.weights[name] = weight

    def store_weights(self, weights):
        '''Store the weights in the weights dictionary.'''
        assert ((type(weights) == dict) | (type(weights) == omegaconf.dictconfig.DictConfig)), f"Weights should be a dictionary."
        for name, weight in weights.items():
            self.store_weight(name, weight)

    @property
    def n_train(self):
        '''Return the number of training events.'''
        return int((1-self.test_size)*len(self.df))

    @property
    def train_mask(self):
        '''Return the training mask according to `test_size`.'''
        # get array of indices of self.df
        indices = np.arange(len(self.df))
        mask = np.ones(len(self.df), dtype=bool)
        return np.where(indices < self.n_train, mask, ~mask)

    @property
    def test_mask(self):
        '''Return the testing mask according to `test_size`.'''
        return ~self.train_mask

    @property
    def train(self):
        '''Return the training dataset according to `test_size`.'''
        return self.df[:self.n_train]

    @property
    def test(self):
        '''Return the testing dataset according to `test_size`.'''
        return self.df[self.n_train:]

    @abstractmethod
    def save(self, output_file):
        """Method to save the dataset. Specific to each dataset type."""
        pass
