#!/bin/bash

# Define the input files replacing tt+B from the inclusive sample to the ttbb one
input_files=(
    "/eos/user/m/mmarcheg/ttHbb/training_datasets/2018/output_ttHTobb_ttToSemiLep_2018.parquet"
    "/eos/user/m/mmarcheg/ttHbb/training_datasets/2018/output_TTbbSemiLeptonic_2018_TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B.parquet"
    "/eos/user/m/mmarcheg/ttHbb/training_datasets/2018/output_TTToSemiLeptonic_2018_TTToSemiLeptonic__TTToSemiLeptonic_tt+C.parquet"
    "/eos/user/m/mmarcheg/ttHbb/training_datasets/2018/output_TTToSemiLeptonic_2018_TTToSemiLeptonic__TTToSemiLeptonic_tt+LF.parquet"
)

output_file="/afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet/datasets/tthbb_ttbar/tthbb_ttbar_with_ttbb_merging_2018.h5"

command="python scripts/dataset/parquet_to_h5.py --cfg parameters/features_spanet_ttbb.yaml -i ${input_files[*]} -o $output_file --reweigh --entrystop 10000"

$command
