import os
import argparse
from collections import defaultdict

import numpy as np
import awkward as ak

import vector
vector.register_numba()
vector.register_awkward()

from omegaconf import OmegaConf

from coffea.util import load
from coffea.processor.accumulator import column_accumulator
from coffea.processor import accumulate

# Read arguments from command line: input file and output directory. Description: script to convert ntuples from coffea file to parquet file.
parser = argparse.ArgumentParser(description='Convert awkward ntuples in coffea files to parquet files.')
parser.add_argument('-c', '--cfg', type=str, required=True, help='YAML configuration file with input features and features to pad')
parser.add_argument('-i', '--input', type=str, required=True, help='Input coffea file')
parser.add_argument('-o', '--output', type=str, required=True, help='Output parquet file')
parser.add_argument('--cat', type=str, default="semilep_LHE", required=False, help='Event category')

args = parser.parse_args()

## Loading the exported dataset
# We open the .coffea file and read the output accumulator. The ntuples for the training are saved under the key `columns`.

if not os.path.exists(args.input):
    raise ValueError(f"Input file {args.input} does not exist.")

# Check if the output file is already existing
if os.path.exists(args.output):
    raise ValueError(f"Output file {args.output} already exists.")

# Check that the format of the output file is `.parquet`
if not args.output.endswith(".parquet"):
    raise ValueError(f"Output file {args.output} should have the `.parquet` extension.")

# Create the output directory if it does not exist
os.makedirs(os.path.dirname(args.output), exist_ok=True)

df = load(args.input)

if not args.cat in df["cutflow"].keys():
    raise ValueError(f"Event category `{args.cat}` not found in the input file.")

# Load the features and features_pad dictionaries with OmegaConf
cfg = OmegaConf.load(args.cfg)
features = cfg["features"]
features_pad = cfg["features_pad"]
awkward_collections = cfg["awkward_collections"]
matched_collections_dict = cfg["matched_collections"]

samples = df["columns"].keys()
print("Samples: ", samples)

for sample in samples:
    
    # Compose the features dictionary with common features and sample-specific features
    features_dict = features["common"].copy()
    if sample in features["by_sample"].keys():
        features_dict.update(features["by_sample"][sample])

    # Compose the dictionary of features to pad
    features_pad_dict = features_pad["common"].copy()
    if sample in features_pad["by_sample"].keys():
        features_pad_dict.update(features_pad["by_sample"][sample])

    # Create a default dictionary of dictionaries to store the arrays
    array_dict = {k : defaultdict(dict) for k in features_dict.keys()}
    datasets = df["columns"][sample].keys()
    print("Datasets: ", datasets)

    ## Normalize the genweights
    # Since the array `weight` is filled on the fly with the weight associated with the event, it does not take into account the overall scaling by the sum of genweights (`sum_genweights`).
    # In order to correct for this, we have to scale by hand the `weight` array dividing by the sum of genweights.
    for dataset in datasets:
        weight = df["columns"][sample][dataset][args.cat]["weight"].value
        weight_new = column_accumulator(weight / df["sum_genweights"][dataset])
        df["columns"][sample][dataset][args.cat]["weight"] = weight_new

    ## Accumulate ntuples from different data-taking eras
    # In order to enlarge our training sample, we merge ntuples coming from different data-taking eras.
    cs = accumulate([df["columns"][sample][dataset][args.cat] for dataset in datasets])

    ## Build the Momentum4D arrays for the jets, partons, leptons, met and higgs
    # In order to get the numpy array from the column_accumulator, we have to access the `value` attribute.
    for collection, variables in features_dict.items():
        for key_feature, key_coffea in variables.items():
            if (collection == "JetGoodMatched") & (key_coffea == "provenance"):
                array_dict[collection][key_feature] = cs[f"PartonMatched_{key_coffea}"].value
            else:
                array_dict[collection][key_feature] = cs[f"{collection}_{key_coffea}"].value
            
        # Add padded features to the array, according to the features dictionary
        if collection in features_pad_dict.keys():
            for key_feature, value in features_pad_dict[collection].items():
                array_dict[collection][key_feature] = value * np.ones_like(cs[f"{collection}_pt"].value)

    # The awkward arrays are zipped together to form the Momentum4D arrays.
    # If the collection is not a Momentum4D array, it is zipped as it is,
    # otherwise the Momentum4D arrays are zipped together and unflattened depending on the number of objects in the collection.
    zipped_dict = defaultdict(dict)
    for collection in array_dict.keys():
        if collection in awkward_collections:
            zipped_dict[collection] = ak.unflatten(ak.zip(array_dict[collection], with_name='Momentum4D'), cs[f"{collection}_N"].value)
        else:
            zipped_dict[collection] = ak.zip(array_dict[collection], with_name='Momentum4D')
        print(f"Collection: {collection}")
        print("Fields: ", zipped_dict[collection].fields)

    for collection in zipped_dict.keys():
        # Pad the matched collections with None if there is no matching
        if collection in matched_collections_dict.keys():
            matched_collection = matched_collections_dict[collection]
            masked_arrays = ak.mask(zipped_dict[matched_collection], zipped_dict[matched_collection].pt==-999, None)
            zipped_dict[matched_collection] = masked_arrays
            # Add the matched flag and the provenance to the matched jets
            if collection == "JetGood":
                is_matched = ~ak.is_none(masked_arrays, axis=1)
                zipped_dict[collection] = ak.with_field(zipped_dict[collection], is_matched, "matched")
                zipped_dict[collection] = ak.with_field(zipped_dict[collection], ak.fill_none(masked_arrays.prov, -1), "prov")

    # Add the remaining keys to the zipped dictionary
    zipped_dict["event"] = ak.zip({"weight" : cs["weight"].value})

    # The Momentum4D arrays are zipped together to form the final dictionary of arrays.
    print("Zipping the collections into a single dictionary...")
    df_out = ak.zip(zipped_dict, depth_limit=1)
    print(f"Saving the output dataset to file: {os.path.abspath(args.output)}")
    ak.to_parquet(df_out, args.output)