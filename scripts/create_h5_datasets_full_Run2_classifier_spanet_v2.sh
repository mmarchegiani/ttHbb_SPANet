#!/bin/bash
STORE_DIR="/eos/user/m/mmarcheg/ttHbb/training_datasets/spanet_v2"
TTHBB_SPANET_DIR=/afs/cern.ch/work/m/mmarcheg/ttHbb/ttHbb_SPANet

input_files=(
    "$STORE_DIR/parquet/output_ttHTobb_ttToSemiLep_2016_PreVFP.parquet"
    "$STORE_DIR/parquet/output_TTbbSemiLeptonic_4f_tt+B_2016_PreVFP.parquet"
    "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+C_2016_PreVFP.parquet"
    "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+LF_2016_PreVFP.parquet"
    "$STORE_DIR/parquet/output_ttHTobb_ttToSemiLep_2016_PostVFP.parquet"
    "$STORE_DIR/parquet/output_TTbbSemiLeptonic_4f_tt+B_2016_PostVFP.parquet"
    "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+C_2016_PostVFP.parquet"
    "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+LF_2016_PostVFP.parquet"
    "$STORE_DIR/parquet/output_ttHTobb_ttToSemiLep_2017.parquet"
    "$STORE_DIR/parquet/output_TTbbSemiLeptonic_4f_tt+B_2017.parquet"
    "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+C_2017.parquet"
    "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+LF_2017.parquet"
    "$STORE_DIR/parquet/output_ttHTobb_ttToSemiLep_2018.parquet"
    "$STORE_DIR/parquet/output_TTbbSemiLeptonic_4f_tt+B_2018.parquet"
    "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+C_2018.parquet"
    "$STORE_DIR/parquet/output_TTToSemiLeptonic_tt+LF_2018.parquet"
)
output_file="$STORE_DIR/h5/tthbb_ttbar_with_ctag_one_hot_encoding_full_Run2.h5"
command="python $TTHBB_SPANET_DIR/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg $TTHBB_SPANET_DIR/parameters/features_spanet_one_hot_encoding.yaml -i ${input_files[*]} -o $output_file"
$command
