import os
import gc
from enum import Enum
from collections import defaultdict

import numpy as np
import awkward as ak
import h5py
import pyarrow.parquet as pq

from .base import Dataset

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

class SPANetDataset(Dataset):
    def __init__(self, input_file, cfg, shuffle=True, reweigh=False, entrystop=None, has_data=False, fully_matched=False, enable_streaming=True):
        # Streaming mode processes one file at a time and never loads all data into RAM.
        # It requires no pre-loaded df, so use lazy=True only when streaming is active
        # and fully_matched is off (fully_matched needs all data in memory to apply its filter).
        self.enable_streaming = enable_streaming
        self.fully_matched = fully_matched
        lazy = enable_streaming
        super().__init__(input_file, cfg, shuffle=shuffle, reweigh=reweigh, entrystop=entrystop, has_data=has_data, lazy=lazy)
        # base.__init__ already calls load_input() when lazy=False, so self.df is set.
        # fully_matched=True forces lazy=False above, so the data is available here.
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
        #update to select only fully matched events for the fully hadronic channel, 
        mask_fullymatched = ak.sum(self.dataset.df[self.collection["Jet"]].matched == True, axis=1) >= 6
        df = self.dataset.df[mask_fullymatched]
        jets = df[self.collection["Jet"]]

        # We require exactly 2 jets from the Higgs, 3 jets from the W or hadronic top, and 1 lepton from the leptonic top
        jets_higgs = jets[jets.prov == 1]
        mask_match = ak.num(jets_higgs) == 2

        #jets_w_thadr = jets[(jets.prov == 5) | (jets.prov == 2)]
        #mask_match = mask_match & (ak.num(jets_w_thadr) == 3)
        jets_w_top = jets[(jets.prov == 5) | (jets.prov == 2)]
        mask_match = mask_match & (ak.num(jets_w_top) == 3)

        #jets_tlep = jets[jets.prov == 3]
        #mask_match = mask_match & (ak.num(jets_tlep) == 1)
        jets_w_antitop = jets[(jets.prov == 6) | (jets.prov == 3)]
        mask_match = mask_match & (ak.num(jets_w_antitop) == 3)

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
            self.file.create_group(f"{SpecialKey.Targets.value}/{target}")
        for object in self.input_features.keys():
            self.file.create_group(f"{SpecialKey.Inputs.value}/{object}")
        for group in self.classification_targets:
            self.file.create_group(f"{SpecialKey.Classifications.value}/{group}")

    def create_targets(self, df):
        jets = df[self.collection["Jet"]]
        indices = ak.local_index(jets)
        n_events = len(df)

        # For mixed classifier datasets (e.g. signal + QCD), provenance may be unavailable.
        # In that case, keep target tensors present but mark all assignments as unmatched.
        if "prov" not in jets.fields:
            for target in self.targets:
                if target == "h":
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b1", (n_events,), dtype='int64', data=-1 * np.ones(n_events, dtype=np.int64))
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b2", (n_events,), dtype='int64', data=-1 * np.ones(n_events, dtype=np.int64))
                elif target == "t1":
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/q1", (n_events,), dtype='int64', data=-1 * np.ones(n_events, dtype=np.int64))
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/q2", (n_events,), dtype='int64', data=-1 * np.ones(n_events, dtype=np.int64))
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/b", (n_events,), dtype='int64', data=-1 * np.ones(n_events, dtype=np.int64))
                elif target == "t2":
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/q1", (n_events,), dtype='int64', data=-1 * np.ones(n_events, dtype=np.int64))
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/q2", (n_events,), dtype='int64', data=-1 * np.ones(n_events, dtype=np.int64))
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/b", (n_events,), dtype='int64', data=-1 * np.ones(n_events, dtype=np.int64))
                else:
                    raise NotImplementedError
            return

        for target in self.targets:
            if target == "h":
                mask = jets.prov == 1 # H->b1b2 
                # We select the local indices of jets matched with the Higgs
                # The indices are padded with None such that there are 2 entries per event
                # The None values are filled with -1 (a nan value).
                indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                index_b1 = indices_prov[:,0]
                index_b2 = indices_prov[:,1]

                self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b1", np.shape(index_b1), dtype='int64', data=index_b1)
                self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b2", np.shape(index_b2), dtype='int64', data=index_b2)

            elif target == "t1":
                mask = jets.prov == 5 # decay quarks from W (from top)  
                indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                index_q1 = indices_prov[:,0]
                index_q2 = indices_prov[:,1]

                mask = jets.prov == 2 # bquarks from top 
                index_b_top= ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]

                self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/q1", np.shape(index_q1), dtype='int64', data=index_q1)
                self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/q2", np.shape(index_q2), dtype='int64', data=index_q2)
                self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/b", np.shape(index_b_top), dtype='int64', data=index_b_top)

            elif target == "t2":
                mask = jets.prov == 6 # non-b decay quarks from  W (from antitop) 
                indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)
                index_q1 = indices_prov[:,0]
                index_q2 = indices_prov[:,1]
                
                mask = jets.prov == 3 # bquarks from antitop 
                index_b_antitop = ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]

                self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/q1", np.shape(index_q1), dtype='int64', data=index_q1)
                self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/q2", np.shape(index_q2), dtype='int64', data=index_q2)
                self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/b", np.shape(index_b_antitop), dtype='int64', data=index_b_antitop)
            else:
                raise NotImplementedError

    def create_classifications(self, df):
        '''Create the classification targets in the h5 file.'''
        for group, targets in self.classification_targets.items():
            for target in targets:
                if group == SpecialKey.Event.value:
                    try:
                        values = df[target]
                    except:
                        raise Exception(f"Target {target} not found in the dataframe.")
                    self.file.create_dataset(f"{SpecialKey.Classifications.value}/{group}/{target}", np.shape(values), dtype='int64', data=values)
                else:
                    raise NotImplementedError

    def create_weights(self, df):
        '''Create the weights in the h5 file.'''
        weights = df.event.weight
        print("Creating dataset: ", f"{SpecialKey.Weights.value}/weight")
        self.file.create_dataset(f"{SpecialKey.Weights.value}/weight", np.shape(weights), dtype='float32', data=weights)

    def create_inputs(self, df):
        '''Create the input arrays in the h5 file.'''
        features = self.get_object_features(df)

        for obj, feats in features.items():
            for feat, val in feats.items():
                if feat == "MASK":
                    dtype = 'bool'
                else:
                    dtype = 'float32'
                dataset_name = f"{SpecialKey.Inputs.value}/{obj}/{feat}"
                print("Creating dataset: ", dataset_name)
                ds = self.file.create_dataset(dataset_name, np.shape(val), dtype=dtype, data=val)

    def get_object_features(self, df):

        df_features = defaultdict(dict)
        for obj, features in self.input_features.items():

            features_dict = {}
            if obj == "Event":
                for feat in features:
                    if feat == "ht":
                        features_dict["ht"] = ak.sum(df["JetGood"]["pt"], axis=1)
                    else:
                        raise NotImplementedError
            else:
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

    def save(self, output_file):
        '''Save the h5 file for both the training and testing datasets.'''
        if self.enable_streaming:
            self._save_streaming(output_file)
        else:
            self._save_in_memory(output_file)

    def _save_in_memory(self, output_file):
        '''Save using the full in-memory dataset (used when fully_matched=True).'''
        for dataset_type in ["train", "test"]:
            print(f"Processing dataset: {dataset_type} ({len(getattr(self, dataset_type))} events)")

            df = getattr(self, dataset_type)
            output_file_custom = output_file.replace(".h5", f"_{dataset_type}_{len(df)}.h5")
            self.check_output(output_file_custom)

            print("Creating output file: ", output_file_custom)
            self.file = h5py.File(output_file_custom, "w")
            self.create_groups()
            if not self.has_data:
                self.create_targets(df)
            self.create_classifications(df)
            self.create_inputs(df)
            self.create_weights(df)
            print(self.file)
            self.print()
            self.file.close()

    def _save_streaming(self, output_file):
        '''Memory-efficient save: processes one parquet file at a time.

        Instead of loading all files into RAM and then writing, this method reads
        event counts from parquet metadata (zero data loading), pre-computes the
        train/test split, then processes each file individually — loading, writing
        to h5, and immediately freeing the data before moving to the next file.

        Shuffle is applied per-file (and file order is randomised when shuffle=True),
        which is a good approximation of a global shuffle without the memory cost.
        '''
        # Count events from parquet metadata without loading any column data.
        file_counts = []
        remaining = self.entrystop
        for f in self.input_files:
            n = pq.read_metadata(f).num_rows
            if remaining is not None:
                n = min(n, remaining)
                remaining -= n
            file_counts.append(n)
            if remaining is not None and remaining <= 0:
                break

        files = self.input_files[:len(file_counts)]

        if self.shuffle:
            order = np.random.permutation(len(files))
            files = [files[i] for i in order]
            file_counts = [file_counts[i] for i in order]

        total = sum(file_counts)
        n_train = int((1 - self.test_size) * total)
        n_test = total - n_train

        print(f"Total events: {total}, train: {n_train}, test: {n_test}")

        train_path = output_file.replace(".h5", f"_train_{n_train}.h5")
        test_path = output_file.replace(".h5", f"_test_{n_test}.h5")
        self.check_output(train_path)
        self.check_output(test_path)

        print(f"Creating output files:")
        print(f"    - {train_path}")
        print(f"    - {test_path}")

        with h5py.File(train_path, "w") as h5_train, \
             h5py.File(test_path, "w") as h5_test:

            for h5 in [h5_train, h5_test]:
                self.file = h5
                self.create_groups()

            train_budget = n_train

            for file_path, max_events in zip(files, file_counts):
                df = self._process_file(file_path, max_events)

                if self.shuffle:
                    df = df[np.random.permutation(len(df))]

                n_file_train = min(train_budget, len(df))
                n_file_test = len(df) - n_file_train
                train_budget -= n_file_train

                if n_file_train > 0:
                    self._append_chunk(h5_train, df[:n_file_train])
                if n_file_test > 0:
                    self._append_chunk(h5_test, df[n_file_train:])

                del df
                gc.collect()

            for h5 in [h5_train, h5_test]:
                print(h5)
                self.file = h5
                self.print()

    def _append_chunk(self, h5_file, df):
        '''Append one chunk of events to h5, creating resizable datasets on first write.'''
        n = len(df)

        def append(name, data, dtype=None):
            arr = np.asarray(data, dtype=dtype)
            if name in h5_file:
                ds = h5_file[name]
                old = ds.shape[0]
                ds.resize(old + n, axis=0)
                ds[old:old + n] = arr
            else:
                maxshape = (None,) + arr.shape[1:]
                h5_file.create_dataset(name, data=arr, maxshape=maxshape, chunks=True)

        # Targets
        if not self.has_data:
            jets = df[self.collection["Jet"]]
            indices = ak.local_index(jets)
            has_prov = "prov" in jets.fields

            for target in self.targets:
                if target == "h":
                    if has_prov:
                        mask = jets.prov == 1
                        idx = ak.fill_none(ak.pad_none(indices[mask], 2), -1)
                        append(f"TARGETS/h/b1", ak.to_numpy(idx[:, 0]), np.int64)
                        append(f"TARGETS/h/b2", ak.to_numpy(idx[:, 1]), np.int64)
                    else:
                        append(f"TARGETS/h/b1", -np.ones(n, dtype=np.int64))
                        append(f"TARGETS/h/b2", -np.ones(n, dtype=np.int64))
                elif target == "t1":
                    if has_prov:
                        idx_q = ak.fill_none(ak.pad_none(indices[jets.prov == 5], 2), -1)
                        idx_b = ak.fill_none(ak.pad_none(indices[jets.prov == 2], 1), -1)[:, 0]
                        append(f"TARGETS/t1/q1", ak.to_numpy(idx_q[:, 0]), np.int64)
                        append(f"TARGETS/t1/q2", ak.to_numpy(idx_q[:, 1]), np.int64)
                        append(f"TARGETS/t1/b",  ak.to_numpy(idx_b), np.int64)
                    else:
                        append(f"TARGETS/t1/q1", -np.ones(n, dtype=np.int64))
                        append(f"TARGETS/t1/q2", -np.ones(n, dtype=np.int64))
                        append(f"TARGETS/t1/b",  -np.ones(n, dtype=np.int64))
                elif target == "t2":
                    if has_prov:
                        idx_q = ak.fill_none(ak.pad_none(indices[jets.prov == 6], 2), -1)
                        idx_b = ak.fill_none(ak.pad_none(indices[jets.prov == 3], 1), -1)[:, 0]
                        append(f"TARGETS/t2/q1", ak.to_numpy(idx_q[:, 0]), np.int64)
                        append(f"TARGETS/t2/q2", ak.to_numpy(idx_q[:, 1]), np.int64)
                        append(f"TARGETS/t2/b",  ak.to_numpy(idx_b), np.int64)
                    else:
                        append(f"TARGETS/t2/q1", -np.ones(n, dtype=np.int64))
                        append(f"TARGETS/t2/q2", -np.ones(n, dtype=np.int64))
                        append(f"TARGETS/t2/b",  -np.ones(n, dtype=np.int64))
                else:
                    raise NotImplementedError

        # Classifications
        for group, targets in self.classification_targets.items():
            for target in targets:
                if group == SpecialKey.Event.value:
                    append(f"CLASSIFICATIONS/{group}/{target}", ak.to_numpy(df[target]), np.int64)
                else:
                    raise NotImplementedError

        # Inputs
        features = self.get_object_features(df)
        for obj, feats in features.items():
            for feat, val in feats.items():
                dtype = bool if feat == "MASK" else np.float32
                append(f"INPUTS/{obj}/{feat}", val, dtype)

        # Weights
        append("WEIGHTS/weight", ak.to_numpy(df.event.weight), np.float32)