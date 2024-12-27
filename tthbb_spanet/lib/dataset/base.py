import os
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod

import numba
import vector
vector.register_awkward()
vector.register_numba()

import numpy as np
import awkward as ak
import h5py

import omegaconf
from omegaconf import OmegaConf

class Dataset:
    def __init__(self, input_file, cfg=None, shuffle=True, reweigh=False, entrystop=None, has_data=False, label=True):
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
        years = ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]
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
            if s == sample:
                df[label] = ak.values_astype(np.ones(len(df), dtype=int), int)
            else:
                df[label] = ak.values_astype(np.zeros(len(df), dtype=int), int)

        # Define one hot encoded label for multiclassifier
        if self.one_hot_encoding:
            for label, mapping in self.mapping_encoding.items():
                df[label] = ak.values_astype(np.ones(len(df), dtype=int) * mapping[sample], int)
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

    def load_input(self):
        '''Load the input file.'''
        # Check if the parquet files exist
        for input_file in self.input_files:
            if not os.path.exists(input_file):
                raise ValueError(f"Input file {input_file} does not exist.")
            if not input_file.endswith(".parquet"):
                raise ValueError(f"Input file {input_file} should have the `.parquet` extension.")
        # Read the parquet files
        print(f"Reading {len(self.input_files)} parquet files: ", self.input_files)
        dfs = []
        for input_file in self.input_files:
            print("Reading file: ", input_file)
            df = ak.from_parquet(input_file)
            # For data, save weights of 1 for all events
            if self.has_data & (not "event" in df.fields):
                df["event"] = ak.zip({"weight": np.ones(len(df), dtype=np.float)})
            # Reweigh the events by a factor
            if self.reweigh:
                df = self.scale_weights(df, self.get_sample_name(input_file))
            # Get sample name from the input file name
            if self.label:
                df = self.build_labels(df, self.get_sample_name(input_file))
                # Add metadata to the dataframe
                df["metadata"] = ak.zip({"year": len(df)*[self.get_year(input_file)]})
                year = df.metadata.year
                # This is needed in order to have a 1D array with one year string per event
                df["metadata"] = ak.with_field(df.metadata, year[:,0], "year")

            dfs.append(df)
        # Return the concatenated dataframe
        # If shuffle is True, the events are randomly shuffled
        df_concat = ak.concatenate(dfs)
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
