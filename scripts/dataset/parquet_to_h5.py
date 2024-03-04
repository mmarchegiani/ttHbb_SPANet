import os
import argparse
from collections import defaultdict

import numpy as np
import awkward as ak
import numba
import h5py

import vector
vector.register_numba()
vector.register_awkward()

from omegaconf import OmegaConf

def create_groups(file):
    file.create_group("TARGETS/t1") # hadronic top -> q1 q2 b
    file.create_group("TARGETS/t2") # leptonic top -> b
    file.create_group("TARGETS/h") # higgs -> b1 b2
    file.create_group("INPUTS")
    file.create_group("INPUTS/Jet")
    file.create_group("INPUTS/Lepton")
    file.create_group("INPUTS/Met")
    file.create_group("INPUTS/Event")
    return file

def create_targets(file, particle, jets):
    indices = ak.local_index(jets)
    
    if particle == "h":
        mask = jets.prov == 1 # H->b1b2
        # We select the local indices of jets matched with the Higgs
        # The indices are padded with None such that there are 2 entries per event
        # The None values are filled with -1 (a nan value).
        indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

        index_b1 = indices_prov[:,0]
        index_b2 = indices_prov[:,1]
        
        file.create_dataset("TARGETS/h/b1", np.shape(index_b1), dtype='int64', data=index_b1)
        file.create_dataset("TARGETS/h/b2", np.shape(index_b2), dtype='int64', data=index_b2)
        
    elif particle == "t1":
        mask = jets.prov == 5 # W->q1q2 from t1
        indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

        index_q1 = indices_prov[:,0]
        index_q2 = indices_prov[:,1]

        mask = jets.prov == 2 # t1->Wb
        index_b_hadr = ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]
                
        file.create_dataset("TARGETS/t1/q1", np.shape(index_q1), dtype='int64', data=index_q1)
        file.create_dataset("TARGETS/t1/q2", np.shape(index_q2), dtype='int64', data=index_q2)
        file.create_dataset("TARGETS/t1/b", np.shape(index_b_hadr), dtype='int64', data=index_b_hadr)
                
    elif particle == "t2":
        mask = jets.prov == 3 # t2->b
        index_b_lep = ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]

        file.create_dataset("TARGETS/t2/b", np.shape(index_b_lep), dtype='int64', data=index_b_lep)

def get_object_features(df, collection=None, features=["pt", "eta", "sin_phi", "cos_phi"]):

    if collection in df.fields:
        objects = df[collection]
    features_dict = {}
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
    elif "ht" in features:
        features_dict["ht"] = ak.sum(df["JetGood"]["pt"], axis=1)
    return features_dict

def create_inputs(file, df):
    features = defaultdict(dict)
    for obj, feats in input_features.items():
        features["Jet"] = get_object_features(df, "JetGood", features=input_features["Jet"])
        features["Lepton"] = get_object_features(df, "LeptonGood", features=input_features["Lepton"])
        features["Met"] = get_object_features(df, "MET", features=input_features["Met"])
        features["Event"] = get_object_features(df, features=input_features["Event"])

    for obj, feats in features.items():
        for feat, val in feats.items():
            if feat == "MASK":
                dtype = 'bool'
            else:
                dtype = 'float32'
            dataset_name = f"INPUTS/{obj}/{feat}"
            print("Creating dataset: ", dataset_name)
            ds = file.create_dataset(dataset_name, np.shape(val), dtype=dtype, data=val)

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))

# Read arguments from command line: input file and output directory. Description: script to convert ntuples from coffea file to parquet file.
parser = argparse.ArgumentParser(description='Convert awkward ntuples in coffea files to parquet files.')
parser.add_argument('-c', '--cfg', type=str, required=True, help='YAML configuration file with input features')
parser.add_argument('-i', '--input', type=str, required=True, nargs='+', help='Input parquet file')
parser.add_argument('-o', '--output', type=str, required=True, help='Output h5 file')
parser.add_argument('-f', '--frac_train', type=float, default=0.8, required=False, help='Fraction of events to be used for training')
parser.add_argument('-fm', '--fully_matched', action='store_true', required=False, help='Use only fully matched events')

args = parser.parse_args()

# Loading the config file
cfg = OmegaConf.load(args.cfg)
input_features = cfg["input"]

# Check if the parquet files exist
for input_file in args.input:
    if not os.path.exists(input_file):
        raise ValueError(f"Input file {input_file} does not exist.")
    if not input_file.endswith(".parquet"):
        raise ValueError(f"Input file {input_file} should have the `.parquet` extension.")
# Check the output file extension
output_dir = os.path.abspath(os.path.dirname(args.output))
filename, file_extension = os.path.splitext(args.output)
if not file_extension == ".h5":
    raise ValueError(f"Output file {args.output} should be in .h5 format.")
# Check if output file exists
if os.path.exists(args.output):
    raise ValueError(f"Output file {args.output} already exists.")
os.makedirs(output_dir, exist_ok=True)

# Read the parquet files
print(f"Reading {len(args.input)} parquet files: ", args.input)
dfs = []
for input_file in args.input:
    print("Reading file: ", input_file)
    df = ak.from_parquet(input_file)
    dfs.append(df)

# Merge the dataframes into a single dataframe
df_all = ak.concatenate(dfs)

# Dictionary of train and test datasets

print("Splitting dataset into train and test...")
df_dict = {
    "train" : df_all[:int(len(df_all)*args.frac_train)],
    "test" : df_all[int(len(df_all)*args.frac_train):]
}

for dataset, df in df_dict.items():
    print(f"Processing dataset: {dataset} ({len(df)} events)")
    # We select only events where all the jets are parton-matched, so that we know their provenance
    if args.fully_matched:
        mask_fullymatched = ak.sum(df.JetGood.matched == True, axis=1) >= 6
        df = df[mask_fullymatched]

        # We require exactly 2 jets from the Higgs, 3 jets from the W or hadronic top, and 1 lepton from the leptonic top
        jets_higgs = df.JetGood[df.JetGood.prov == 1]
        mask_match = ak.num(jets_higgs) == 2

        jets_w_thadr = df.JetGood[(df.JetGood.prov == 5) | (df.JetGood.prov == 2)]
        mask_match = mask_match & (ak.num(jets_w_thadr) == 3)

        jets_tlep = df.JetGood[df.JetGood.prov == 3]
        mask_match = mask_match & (ak.num(jets_tlep) == 1)

        df = df[mask_match]
        print(f"Selected {len(df)} fully matched events")
    else:
        print(f"Selected {len(df)} events")

    output_file = args.output.replace(".h5", f"_{dataset}_{len(df)}.h5")
    print("Creating output file: ", output_file)
    with h5py.File(output_file, "w") as f:
        f = create_groups(f)
        # Create targets in the file
        for particle in ["h", "t1", "t2"]:
            create_targets(f, particle, df.JetGood)
        # Create input arrays in the files
        create_inputs(f, df)
        print(f)
        h5_tree(f)