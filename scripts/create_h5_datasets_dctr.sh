#!/bin/bash

datasets_data=(DATA_SingleEle DATA_SingleMuon)
#years=("2018" "2017" "2016_PostVFP" "2016_PreVFP")
years=("2018")
for year in ${years[@]};
do
    if [ $year == "2016_PreVFP" ]
    then
        eras=(EraB EraC EraD EraE EraF)
    elif [ $year == "2016_PostVFP" ]
    then
        eras=(EraF EraG EraH)
    elif [ $year == "2017" ]
    then
        eras=(EraB EraC EraD EraE EraF)
    elif [ $year == "2018" ]
    then
        eras=(EraA EraB EraC EraD)
    fi
    # Define the input files replacing tt+B from the inclusive sample to the ttbb one
    input_files=(
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_ttHTobb_${year}.parquet"
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_TTTo2L2Nu_${year}.parquet"
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_TTbbSemiLeptonic_4f_tt+B_${year}.parquet"
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_TTToSemiLeptonic_tt+C_${year}.parquet"
        "/eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_TTToSemiLeptonic_tt+LF_${year}.parquet"
    )
    # Append to input_files the data files
    for dataset_data in "${datasets_data[@]}"
    do
        for era in "${eras[@]}"
        do
            input_files+=("/eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year}/output_${dataset_data}_${year}_${era}.parquet")
        done
    done
    output_file="/eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/h5/2018/tthbb_ttbar_DATA_with_ttbb_merging_ttbb_reweighed_${year}.h5"
    command="python scripts/dataset/parquet_to_h5.py --cfg parameters/features_spanet_dctr.yaml -i ${input_files[*]} -o $output_file --data"
    $command
done
