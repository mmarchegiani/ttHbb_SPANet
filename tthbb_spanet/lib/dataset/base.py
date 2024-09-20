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

# Inherited from https://github.com/Alexanders101/SPANet/blob/master/spanet/dataset/types.py
class SpecialKey(str, Enum):
    Mask = "MASK"
    Event = "EVENT"
    Inputs = "INPUTS"
    Targets = "TARGETS"
    Particle = "PARTICLE"
    Regressions = "REGRESSIONS"
    Permutations = "PERMUTATIONS"
    Classifications = "CLASSIFICATIONS"
    Embeddings = "EMBEDDINGS"
    Weights = "WEIGHTS"

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

class SPANetDataset(Dataset):
    def __init__(self, input_file, output_file, cfg, shuffle=True, reweigh=False, entrystop=None, has_data=False, fully_matched=False):
        super().__init__(input_file, output_file, cfg, shuffle=True, reweigh=False, entrystop=None, has_data=False)
        self.fully_matched = fully_matched
        if self.fully_matched:
            self.select_fully_matched()

    def load_config(self):
        '''Load the config file with OmegaConf and read the input features for the SPANet training.'''
        super().load_config()
        self.input_features = self.cfg["input"]
        self.collection = self.cfg["collection"]
        self.targets = self.cfg["particles"]
        self.classification_targets = self.cfg["classification"]

    def select_fully_matched(self):
        '''Select only fully matched events.'''
        mask_fullymatched = ak.sum(self.dataset.df[self.collection["Jet"]].matched == True, axis=1) >= 6
        df = self.dataset.df[mask_fullymatched]
        jets = df[self.collection["Jet"]]

        # We require exactly 2 jets from the Higgs, 3 jets from the W or hadronic top, and 1 lepton from the leptonic top
        jets_higgs = jets[jets.prov == 1]
        mask_match = ak.num(jets_higgs) == 2

        jets_w_thadr = jets[(jets.prov == 5) | (jets.prov == 2)]
        mask_match = mask_match & (ak.num(jets_w_thadr) == 3)

        jets_tlep = jets[jets.prov == 3]
        mask_match = mask_match & (ak.num(jets_tlep) == 1)

        df = df[mask_match]
        print(f"Selected {len(df)} fully matched events")

        self.dataset.df = df

    def check_output(self, output_file):
        '''Check the output file extension and if it already exists.'''
        # Check the output file extension
        filename, file_extension = os.path.splitext(output_file)
        if not file_extension == ".h5":
            raise ValueError(f"Output file {output_file} should be in .h5 format.")
        # Check if output file exists
        if os.path.exists(output_file):
            raise ValueError(f"Output file {output_file} already exists.")
        os.makedirs(os.path.abspath(os.path.dirname(output_file)), exist_ok=True)

    def create_groups(self):
        '''Create the groups in the h5 file.'''
        for target in self.targets:
            self.file.create_group(f"{SpecialKey.Targets}/{target}")
        for object in self.input_features.keys():
            self.file.create_group(f"{SpecialKey.Inputs}/{object}")
        for group in self.classification_targets:
            self.file.create_group(f"{SpecialKey.Classifications}/{group}")

    def create_targets(self, df):
        jets = df[self.collection["Jet"]]
        indices = ak.local_index(jets)

        for target in self.targets:
            if target == "h":
                mask = jets.prov == 1 # H->b1b2
                # We select the local indices of jets matched with the Higgs
                # The indices are padded with None such that there are 2 entries per event
                # The None values are filled with -1 (a nan value).
                indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                index_b1 = indices_prov[:,0]
                index_b2 = indices_prov[:,1]

                self.file.create_dataset(f"{SpecialKey.Targets}/h/b1", np.shape(index_b1), dtype='int64', data=index_b1)
                self.file.create_dataset(f"{SpecialKey.Targets}/h/b2", np.shape(index_b2), dtype='int64', data=index_b2)

            elif target == "t1":
                mask = jets.prov == 5 # W->q1q2 from t1
                indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                index_q1 = indices_prov[:,0]
                index_q2 = indices_prov[:,1]

                mask = jets.prov == 2 # t1->Wb
                index_b_hadr = ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]

                self.file.create_dataset(f"{SpecialKey.Targets}/t1/q1", np.shape(index_q1), dtype='int64', data=index_q1)
                self.file.create_dataset(f"{SpecialKey.Targets}/t1/q2", np.shape(index_q2), dtype='int64', data=index_q2)
                self.file.create_dataset(f"{SpecialKey.Targets}/t1/b", np.shape(index_b_hadr), dtype='int64', data=index_b_hadr)

            elif target == "t2":
                mask = jets.prov == 3 # t2->b
                index_b_lep = ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]

                self.file.create_dataset(f"{SpecialKey.Targets}/t2/b", np.shape(index_b_lep), dtype='int64', data=index_b_lep)
            else:
                raise NotImplementedError

    def create_classifications(self, df):
        '''Create the classification targets in the h5 file.'''
        for group, targets in self.classification_targets.items():
            for target in targets:
                if group == SpecialKey.Event:
                    try:
                        values = df[target]
                    except:
                        raise Exception(f"Target {target} not found in the dataframe.")
                    self.file.create_dataset(f"{SpecialKey.Classifications}/{group}/{target}", np.shape(values), dtype='int64', data=values)
                else:
                    raise NotImplementedError

    def create_weights(self, df):
        '''Create the weights in the h5 file.'''
        weights = df.event.weight
        print("Creating dataset: ", f"{SpecialKey.Weights}/weight")
        self.file.create_dataset(f"{SpecialKey.Weights}/weight", np.shape(weights), dtype='float32', data=weights)

    def create_inputs(self, df):
        '''Create the input arrays in the h5 file.'''
        features = self.get_object_features(df)

        for obj, feats in features.items():
            for feat, val in feats.items():
                if feat == "MASK":
                    dtype = 'bool'
                else:
                    dtype = 'float32'
                dataset_name = f"{SpecialKey.Inputs}/{obj}/{feat}"
                print("Creating dataset: ", dataset_name)
                ds = self.file.create_dataset(dataset_name, np.shape(val), dtype=dtype, data=val)

    def get_object_features(self, df):

        df_features = defaultdict(dict)
        for obj, features in self.input_features.items():

            features_dict = {}
            collection = self.collection[obj]

            if (collection in df.fields):
                objects = df[collection]
                for feat in features:
                    if feat == "MASK":
                        continue
                    if feat in ["sin_phi", "cos_phi"]:
                        phi = objects["phi"]
                        if feat == "sin_phi":
                            values = np.sin(phi)
                        elif feat == "cos_phi":
                            values = np.cos(phi)
                    elif feat == "is_electron":
                        values = ak.values_astype(abs(objects["pdgId"]) == 11, int)
                    else:
                        values = objects[feat]
                    if objects.ndim == 1:
                        features_dict[feat] = ak.to_numpy(values)
                    elif objects.ndim == 2:
                        features_dict[feat] = ak.to_numpy(ak.fill_none(ak.pad_none(values, 16, clip=True), 0))
                    else:
                        raise NotImplementedError

                if "MASK" in features:
                    if not "pt" in features:
                        raise NotImplementedError
                    features_dict["MASK"] = ~(features_dict["pt"] == 0)
            else:
                raise ValueError(f"Collection {collection} not found in the parquet file.")
            df_features[obj] = features_dict
        return df_features

    def h5_tree(self, val, pre=''):
        items = len(val)
        for key, val in val.items():
            items -= 1
            if items == 0:
                # the last item
                if type(val) == h5py._hl.group.Group:
                    print(pre + '└── ' + key)
                    self.h5_tree(val, pre+'    ')
                else:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
            else:
                if type(val) == h5py._hl.group.Group:
                    print(pre + '├── ' + key)
                    self.h5_tree(val, pre+'│   ')
                else:
                    print(pre + '├── ' + key + ' (%d)' % len(val))

    def print(self):
        '''Print the h5 file tree.'''
        self.h5_tree(self.file)

    def save_h5(self, output_file, dataset_type):
        '''Save the h5 file.'''
        assert dataset_type in ["train", "test"], f"Dataset type `{dataset_type}` not recognized."

        self.check_output(output_file)

        df = getattr(self.dataset, dataset_type)
        output_file = output_file.replace(".h5", f"_{dataset_type}_{len(df)}.h5")
        
        print("Creating output file: ", output_file)
        self.file = h5py.File(output_file, "w")
        self.create_groups()
        if not self.has_data:
            self.create_targets(df)
        self.create_classifications(df)
        self.create_inputs(df)
        self.create_weights(df)
        print(self.file)
        self.print()
        self.file.close()

    def save_h5_all(self):
        '''Save the h5 file for both the training and testing datasets.'''
        for dataset in ["train", "test"]:
            print(f"Processing dataset: {dataset} ({len(getattr(self.dataset, dataset))} events)")
            self.save_h5(dataset)