#!/bin/bash
STORE_DIR="/eos/user/m/mmarcheg/ttHbb/training_datasets/spanet_v2"
TTHBB_SPANET_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet

for YEAR in 2016_PreVFP 2016_PostVFP 2017 2018
do
    input_files=(
        "$STORE_DIR/parquet/output_ttHTobb_ttToSemiLep_${YEAR}.parquet"
        "$STORE_DIR/parquet/output_TTbbSemiLeptonic_4f_tt+B_${YEAR}.parquet"
        "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+C_${YEAR}.parquet"
        "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+LF_${YEAR}.parquet"
    )
    output_file="$STORE_DIR/h5/tthbb_ttbar_with_ctag_one_hot_encoding_${YEAR}.h5"
    command="python $TTHBB_SPANET_DIR/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg $TTHBB_SPANET_DIR/parameters/features_spanet_one_hot_encoding.yaml -i ${input_files[*]} -o $output_file"
    $command
done
