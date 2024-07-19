#!/bin/bash

# Define the input files replacing tt+B from the inclusive sample to the ttbb one
years=("2018" "2017" "2016_PostVFP" "2016_PreVFP")
for year in ${years[@]};
do
    input_files=(
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/${year}/output_ttHTobb_ttToSemiLep_${year}.parquet"
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/${year}/output_TTbbSemiLeptonic_${year}_TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B.parquet"
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/${year}/output_TTToSemiLeptonic_${year}_TTToSemiLeptonic__TTToSemiLeptonic_tt+C.parquet"
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/${year}/output_TTToSemiLeptonic_${year}_TTToSemiLeptonic__TTToSemiLeptonic_tt+LF.parquet"
    )
    output_file="/eos/user/m/mmarcheg/ttHbb/training_datasets/classifier/multiclassifier_full_Run2_ttbb_reweighed/tthbb_ttbar_with_ttbb_merging_ttbb_reweighed_${year}.h5"
    command="python scripts/dataset/parquet_to_h5.py --cfg parameters/features_spanet_ttbb.yaml -i ${input_files[*]} -o $output_file"
    $command
done
