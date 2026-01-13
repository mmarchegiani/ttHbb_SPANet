#!/bin/bash

FOLDER="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/h5_files"

# =================================================================================
#   If you want to specify only a subset of files 
#   into FOLDER write the list here and uncomment
# =================================================================================

#FILES_TRAIN=(
#    "ttbb_tthbb_non_semilep_2022_preEE_train_378679.h5"
#    "ttbb_tthbb_non_semilep_2022_postEE_train_1550140.h5"
#    "ttbb_tthbb_non_semilep_2023_preBPix_train_1027407.h5"
#    "ttbb_tthbb_non_semilep_2023_postBPix_train_535243.h5"
#    #"ttbb_tthbb_non_semilep_2024_train_4881159.h5"
#    "tthbb_semilep_genmatched_2022_preEE_train_117517.h5"
#    "tthbb_semilep_genmatched_2022_postEE_train_384913.h5"
#    "tthbb_semilep_genmatched_2023_preBPix_train_413093.h5"
#    "tthbb_semilep_genmatched_2023_postBPix_train_210367.h5"
#    #"tthbb_semilep_genmatched_2024_train_132453.h5"
#    
#)
#FILES_TEST=(
#    "ttbb_tthbb_non_semilep_2022_preEE_test_94670.h5"
#    "ttbb_tthbb_non_semilep_2022_postEE_test_387535.h5"
#    "ttbb_tthbb_non_semilep_2023_preBPix_test_256852.h5"
#    "ttbb_tthbb_non_semilep_2023_postBPix_test_133811.h5"
#    #"ttbb_tthbb_non_semilep_2024_test_1220290.h5"
#    "tthbb_semilep_genmatched_2022_preEE_test_29380.h5"
#    "tthbb_semilep_genmatched_2022_postEE_test_96229.h5"
#    "tthbb_semilep_genmatched_2023_preBPix_test_103274.h5"
#    "tthbb_semilep_genmatched_2023_postBPix_test_52592.h5"
#    #"tthbb_semilep_genmatched_2024_test_33114.h5"
#)
#
## Build full input paths
#INPUTS_TRAIN=()
#INPUTS_TEST=()
#for file in "${FILES_TRAIN[@]}"; do
#  INPUTS_TRAIN+=("${FOLDER}/${file}")
#done
#for file in "${FILES_TEST[@]}"; do
#  INPUTS_TEST+=("${FOLDER}/${file}")
#done

# ====================================================================================

INPUTS_TRAIN=("${FOLDER}"/*train*.h5)
INPUTS_TEST=("${FOLDER}"/*test*.h5)

OUTPUT_TRAIN="${FOLDER}/ttbb_tthbb_non_semilep_tthbb_semilep_genmatched_22_23only_train.h5"
OUTPUT_TEST="${FOLDER}/ttbb_tthbb_non_semilep_tthbb_semilep_genmatched_22_23only_test.h5"

python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/merge_h5_files.py \
  -i "${INPUTS_TRAIN[@]}" \
  -o "${OUTPUT_TRAIN}"

python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/merge_h5_files.py \
  -i "${INPUTS_TEST[@]}" \
  -o "${OUTPUT_TEST}"