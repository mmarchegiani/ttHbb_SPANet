#!/bin/bash

BASE_DIR=/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/parquet_overall

SAMPLES_2022_23=(
        "TWminus" 
        "TWplus" 
        "TBbarQ_tchannel"
        "TbarBQ_tchannel"
        "TBbarLNuB_schannel"
        "TbarBLNuB_schannel"
        "TTNuNu" 
        "TTLNu" 
        "TTZ"
    ) 
    # WW, ZZ, WZ
    #"WJetsToLNu" 
    #"DYJetsToLL_10_50"
    #"DYJetsToLL_50" 
SAMPLES_2024=(
        "TWminus" 
        "TWplus"
        "TTLL_ML_to50" 
        "TTLL_ML_50" 
    )
    #"DYto2L-PtLL_100_200_1J_PtLL_100_200_1J" 
    #"DYto2L-PtLL_200_400_2J_PtLL_200_400_2J" 
    #"DYto2L-PtLL_400_600_1J_PtLL_400_600_1J" 
    #"DYto2L-PtLL_400_600_2J_PtLL_400_600_2J" 
    #"DYto2L-PtLL_600_1J_PtLL_600_1J" 
    #"DYto2L-PtLL_600_2J_PtLL_600_2J" 
    TTBAR=("TTTo2L2Nu" "TTToLNu2Q" "TTBBtoLNu2Q_4FS")
    #"TTHtoNon2B"
SUBSAMPLES=("ttB" "ttC" "ttLF")


for year in "2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix";
do
    input_files=()
    genmatched_input_files=(
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/parquet_overall/TTH_Hto2B_genmatching_${year}.parquet"
    )
    # Check each file and add to array if it exists
    for sample in ${SAMPLES_2022_23[@]};
    do
        file="$BASE_DIR/${sample}_${year}.parquet"
        input_files+=("$file")
    done
    for sample_ttbar in ${TTBAR[@]}; do
        for subsample in "${SUBSAMPLES[@]}"; do
            file_subsample="$BASE_DIR/${sample_ttbar}_${subsample}_${year}.parquet"
            input_files+=("$file_subsample")
        done
    done

    output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/h5_files/ttbb_tthbb_non_semilep_${year}.h5"
    command="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${input_files[*]} -o $output_file"
    $command
    
    genmatched_output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/h5_files/tthbb_semilep_genmatched_${year}.h5"
    command_gen="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${genmatched_input_files[*]} -o $genmatched_output_file"
    $command_gen
done

for year in "2024";
do
    input_files=()
    genmatched_input_files=(
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/parquet_overall/TTH_Hto2B_genmatching_${year}.parquet"
    )
    # Check each file and add to array if it exists
    for sample in ${SAMPLES_2024[@]};
    do
        file="$BASE_DIR/${sample}_${year}.parquet"
        input_files+=("$file")
    done
    for sample_ttbar in ${TTBAR[@]}; do
        for subsample in "${SUBSAMPLES[@]}"; do
            file_subsample="$BASE_DIR/${sample_ttbar}_${subsample}_${year}.parquet"
            input_files+=("$file_subsample")
        done
    done

    output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/h5_files/ttbb_tthbb_non_semilep_${year}.h5"
    command="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${input_files[*]} -o $output_file"
    $command
    
    genmatched_output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/h5_files/tthbb_semilep_genmatched_${year}.h5"
    command_gen="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${genmatched_input_files[*]} -o $genmatched_output_file"
    $command_gen
done
