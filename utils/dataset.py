import os
from enum import Enum
from collections import defaultdict

import numba
import vector
vector.register_awkward()
vector.register_numba()

import numpy as np
import awkward as ak
import h5py

from omegaconf import OmegaConf

from coffea.util import load
from coffea.processor.accumulator import column_accumulator
from coffea.processor import accumulate

mapping_sample = {
    "ttHTobb": 1,
    "TTToSemiLeptonic": 0,
}

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

class ParquetDataset:
    def __init__(self, input_file, output_file, cfg, cat="semilep_LHE", input_ntuples=None):
        self.input_file = input_file
        self.output_file = output_file
        self.cfg = cfg
        self.cat = cat
        self.input_ntuples = input_ntuples

        self.load_config()
        self.load_input()
        self.check_output()
        self.normalize_genweights()
        self.load_features()
        if self.schema == "coffea":
            self.build_arrays_from_accumulator()
        elif self.schema == "parquet":
            self.build_arrays_from_ntuples()
        else:
            raise ValueError(f"Schema {self.schema} not recognized.")

    @property
    def nevents(self):
        if self.schema == "coffea":
            pass
        elif self.schema == "parquet":
            return len(self.events)

    def get_datasets(self, sample=None):
        if self.schema == "coffea":
            return list(self.columns[sample].keys())
        elif self.schema == "parquet":
            return list(self.cutflow.keys())
        else:
            raise ValueError(f"Schema {self.schema} not recognized.")

    @property
    def samples(self):
        if self.schema == "coffea":
            return list(self.columns.keys())
        elif self.schema == "parquet":
            samples = []
            datasets = list(self.cutflow.keys())
            assert len(datasets) == 1, "Multiple datasets found in the datasets metadata."
            dataset = datasets[0]
            for sample, nevt in self.cutflow[dataset].items():
                if nevt == self.nevents:
                    samples.append(sample)
            if len(samples) == 0:
                raise ValueError(f"No sample found with {self.nevents} events.")
            return samples

    def load_input(self):
        '''Load the input file and check if the event category `cat` is present.'''
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file {self.input_file} does not exist.")
        else:
            print(f"Reading input file: {self.input_file}")
            self.df = load(self.input_file)
            self.columns = self.df["columns"]
            if not self.cat in self.df["cutflow"].keys():
                raise ValueError(f"Event category `{self.cat}` not found in the input file.")
            self.sum_genweights = self.df["sum_genweights"]
            self.cutflow = self.df["cutflow"][self.cat]

        if self.input_ntuples:
            self.schema = "parquet"
            print("Reading input ntuples: ", self.input_ntuples)
            self.events = ak.from_parquet(self.input_ntuples)
            nevents = len(self.events)
        else:
            self.schema = "coffea"

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
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)

    def get_weight(self, sample, dataset):
        if self.schema == "coffea":
            weight = self.columns[sample][dataset][self.cat]["weight"].value
        elif self.schema == "parquet":
            weight = self.events.weight
        return weight

    def normalize_genweights(self):
        '''Since the array `weight` is filled on the fly with the weight associated with the event, 
        it does not take into account the overall scaling by the sum of genweights (`sum_genweights`).
        In order to correct for this, we have to scale by hand the `weight` array dividing by the sum of genweights.'''

        for sample in self.samples:
            for dataset in self.get_datasets(sample):
                weight = self.get_weight(sample, dataset)
                if self.schema == "coffea":
                    weight_new = column_accumulator(weight / self.sum_genweights[dataset])
                    self.columns[sample][dataset][self.cat]["weight"] = weight_new
                elif self.schema == "parquet":
                    self.events.weight = weight / self.sum_genweights[dataset]

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

    def build_arrays_from_ntuples(self):
        '''Build the Momentum4D arrays for the jets, partons, leptons, met and higgs, reading from unflattened ntuples.'''

        print("Samples: ", self.samples)

        for sample in self.samples:

            # Build the array_dict from the unflattened ntuples read from parquet
            for collection, variable in self.features_dict[sample].items():
                for key_feature, key_coffea in variable.items():
                    if (collection == "JetGoodMatched") & (key_coffea == "provenance"):
                        self.array_dict[sample][collection][key_feature] = self.events[f"PartonMatched_{key_coffea}"]
                    else:
                        self.array_dict[sample][collection][key_feature] = self.events[f"{collection}_{key_coffea}"]

                # Add padded features to the array, according to the features dictionary
                if collection in self.features_pad_dict[sample].keys():
                    for key_feature, value in self.features_pad_dict[sample][collection].items():
                        self.array_dict[sample][collection][key_feature] = value * ak.ones_like(self.events[f"{collection}_pt"])

            # The awkward arrays are zipped together to form the Momentum4D arrays.
            # If the collection is not a Momentum4D array, it is zipped as it is.
            # Here we assume that the ntuples are unflattened, therefore we don't need to unflatten them before zipping.
            for collection in self.array_dict[sample].keys():
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
            self.zipped_dict[sample]["event"] = ak.zip({"weight" : self.events.weight})

    def build_arrays_from_accumulator(self):
        '''Build the Momentum4D arrays for the jets, partons, leptons, met and higgs.'''
        
        print("Samples: ", self.samples)

        for sample in self.samples:
            datasets = self.get_datasets(sample)
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
            print(f"Saving the output dataset to file: {output_file}")
            ak.to_parquet(df_out, output_file)

class H5Dataset:
    def __init__(self, input_file, output_file, cfg, fully_matched=False, shuffle=True):
        # Load several input files into a list
        if type(input_file) == str:
            self.input_files = [input_file]
        elif type(input_file) == list:
            self.input_files = input_file
        else:
            raise ValueError(f"Input file {input_file} should be a string or a list of strings.")
        self.output_file = output_file
        self.cfg = cfg
        self.fully_matched = fully_matched
        self.shuffle = shuffle

        self.sample_dict = defaultdict(dict)

        self.load_config()
        self.check_output()
        self.dataset = Dataset(self)
        if self.fully_matched:
            self.select_fully_matched()

    @staticmethod
    def get_sample_name(input_file):
        '''Get the sample name from the input file name.'''
        sample_list = []
        for sample in mapping_sample.keys():
            if sample in input_file:
                sample_list.append(sample)
        if len(sample_list) == 0:
            raise ValueError(f"Sample name not found in the input file name: {input_file}.\nAvailable samples: {mapping_sample.keys()}")
        elif len(sample_list) > 1:
            raise ValueError(f"""Multiple sample names found in the input file name: {input_file}.
                                A single sample name should be specified in the file name.\n
                                Available samples: {mapping_sample.keys()}""")
        return sample_list[0]

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
            # Get sample name from the input file name
            df["signal"] = ak.values_astype(mapping_sample[self.get_sample_name(input_file)] * np.ones(len(df), dtype=int), int)
            dfs.append(df)
        # Return the concatenated dataframe
        # If shuffle is True, the events are randomly shuffled
        df_concat = ak.concatenate(dfs)
        if self.shuffle:
            return df_concat[np.random.permutation(len(df_concat))]
        else:
            return df_concat

    def load_config(self):
        '''Load the config file with OmegaConf and read the input features.'''
        self.cfg = OmegaConf.load(self.cfg)
        self.input_features = self.cfg["input"]
        self.collection = self.cfg["collection"]
        self.targets = self.cfg["particles"]
        self.classification_targets = self.cfg["classification"]
        self.frac_train = self.cfg["frac_train"]

    def check_output(self):
        '''Check the output file extension and if it already exists.'''
        # Check the output file extension
        filename, file_extension = os.path.splitext(self.output_file)
        if not file_extension == ".h5":
            raise ValueError(f"Output file {self.output_file} should be in .h5 format.")
        # Check if output file exists
        if os.path.exists(self.output_file):
            raise ValueError(f"Output file {self.output_file} already exists.")
        os.makedirs(os.path.abspath(os.path.dirname(self.output_file)), exist_ok=True)

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
                    if target == "signal":
                        values = df["signal"]
                    else:
                        raise NotImplementedError
                    self.file.create_dataset(f"{SpecialKey.Classifications}/{group}/{target}", np.shape(values), dtype='int64', data=values)
                else:
                    raise NotImplementedError

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
            if obj == "Event":
                if "ht" in features:
                    features_dict["ht"] = ak.sum(df["JetGood"]["pt"], axis=1)
                else:
                    raise NotImplementedError
            else:
                collection = self.collection[obj]

                if (collection in df.fields) & (collection != "Event"):
                    objects = df[collection]
                    for feat in features:
                        if feat in ["MASK", "ht"]: continue
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

    def save_h5(self, dataset_type):
        '''Save the h5 file.'''
        assert dataset_type in ["train", "test"], f"Dataset type `{dataset_type}` not recognized."

        df = getattr(self.dataset, dataset_type)
        output_file = self.output_file.replace(".h5", f"_{dataset_type}_{len(df)}.h5")
        
        print("Creating output file: ", output_file)
        self.file = h5py.File(output_file, "w")
        self.create_groups()
        self.create_targets(df)
        self.create_classifications(df)
        self.create_inputs(df)
        print(self.file)
        self.print()
        self.file.close()

    def save_h5_all(self):
        '''Save the h5 file for both the training and testing datasets.'''
        for dataset in ["train", "test"]:
            print(f"Processing dataset: {dataset} ({len(getattr(self.dataset, dataset))} events)")
            self.save_h5(dataset)

class Dataset:
    def __init__(self, h5_dataset : H5Dataset):
        self.df = h5_dataset.load_input()
        self.frac_train = h5_dataset.frac_train

    @property
    def train(self):
        '''Return the training dataset according to the fraction `frac_train`.'''
        return self.df[:int(self.frac_train*len(self.df))]

    @property
    def test(self):
        '''Return the testing dataset according to the fraction `frac_train`.'''
        return self.df[int(self.frac_train*len(self.df)):]
