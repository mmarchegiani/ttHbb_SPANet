#!/bin/bash
BASE_DIR=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet
RUN_DIR=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/training_fixedwp_dec25/training_no_toppt_var/fixed_wp
EOS_DIR=/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp
MINORBK_2022_23=("TWminus" "TWplus" "TbarBLNuB_schannel" "TbarBQ_tchannel" "TBbarLNuB_schannel" "TBbarQ_tchannel" "WJetsToLNu" "DYto2L-2Jets_MLL-10to50" "DYto2L-2Jets_MLL-50" "TTLL_ML_to50" "TTLL_ML_50" "TTNuNu" "TTLNu" "TTZ") # WW, ZZ, WZ
MINORBK_2024=("TWminus" "TWplus" "WJetsToLNu" "DYto2L-PtLL_100_200_1J_PtLL_100_200_1J" "DYto2L-PtLL_200_400_2J_PtLL_200_400_2J" "DYto2L-PtLL_400_600_1J_PtLL_400_600_1J" "DYto2L-PtLL_400_600_2J_PtLL_400_600_2J" "DYto2L-PtLL_600_1J_PtLL_600_1J" "DYto2L-PtLL_600_2J_PtLL_600_2J" "TTLL_ML_to50" "TTLL_ML_50")
TTBAR=("TTBBtoLNu2Q_4FS")
#TTBAR=("TTTo2L2Nu" "TTToLNu2Q" "TTBBtoLNu2Q_4FS")
#"TTHtoNon2B")
SUBSAMPLES=("ttB" "ttC" "ttLF")
CAT=baseline

# Save ttHTobb_ttToSemiLep datasets to parquet
for YEAR in 2023_postBPix
#for YEAR in 2022_preEE 2022_postEE 2023_preBPix 2023_postBPix
do
    # ttHbb sample divided in semileptonic (containing genmatching) and non_semilep 
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_22_23.yaml -i $RUN_DIR/output_coffea_non_semilep/output_TTH_Hto2B_${YEAR}.coffea -o $EOS_DIR/parquet_overall/TTH_Hto2B_nonsemilep_${YEAR}.parquet --dataset TTH_Hto2B_${YEAR} --ntuples $EOS_DIR/non_semilep_ttHbb/TTH_Hto2B_${YEAR}/non_semilep/baseline --cat ${CAT}
    #python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_genmatched_22_23.yaml -i $RUN_DIR/output_coffea_semilep/output_TTH_Hto2B_${YEAR}.coffea -o $EOS_DIR/parquet_overall/TTH_Hto2B_genmatching_${YEAR}.parquet --dataset TTH_Hto2B_${YEAR} --ntuples $EOS_DIR/semilep_ttHbb_genmatched/TTH_Hto2B_${YEAR}/baseline --cat ${CAT}
    #for SAMPLE in "${MINORBK_2022_23[@]}"; do
    #    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_22_23.yaml -i $RUN_DIR/output_coffea_non_semilep/output_${SAMPLE}_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${SAMPLE}_${YEAR}.parquet --dataset ${SAMPLE}_${YEAR} --ntuples $EOS_DIR/non_semilep_ttHbb/${SAMPLE}_${YEAR}/baseline --cat ${CAT}
    #done
    # ttbar semilepton, dilepton, 4FS and tthnon2b are split in subsamples ttLF, ttB, ttC
    for SAMPLE_TTBAR in "${TTBAR[@]}"; do
        for SUBSAMPLE in "${SUBSAMPLES[@]}"; do
            python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_22_23.yaml -i $RUN_DIR/output_coffea_non_semilep/output_${SAMPLE_TTBAR}_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${SAMPLE_TTBAR}_${SUBSAMPLE}_${YEAR}.parquet --dataset ${SAMPLE_TTBAR}_${YEAR} --ntuples $EOS_DIR/non_semilep_ttHbb/${SAMPLE_TTBAR}_${YEAR}/${SUBSAMPLE}/baseline --cat ${CAT}
        done
    done
done

# separate 2024 because it has different variables (btagging algorithm)
#for YEAR in 2024
#do
    # ttHbb sample divided in semileptonic (containing genmatching) and non_semilep
#    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_2024.yaml -i $RUN_DIR/output_coffea_non_semilep/output_TTH_Hto2B_${YEAR}.coffea -o $EOS_DIR/parquet_overall/TTH_Hto2B_nonsemilep_${YEAR}.parquet --dataset TTH_Hto2B_${YEAR} --ntuples $EOS_DIR/non_semilep_ttHbb/TTH_Hto2B_${YEAR}/non_semilep/baseline --cat ${CAT}
#    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_genmatched_2024.yaml -i $RUN_DIR/output_coffea_semilep/output_TTH_Hto2B_${YEAR}.coffea -o $EOS_DIR/parquet_overall/TTH_Hto2B_genmatching_${YEAR}.parquet --dataset TTH_Hto2B_${YEAR} --ntuples $EOS_DIR/semilep_ttHbb_genmatched/TTH_Hto2B_${YEAR}/baseline --cat ${CAT}
#    for SAMPLE in "${MINORBK_2024[@]}"; do
#        python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_2024.yaml -i $RUN_DIR/output_coffea_non_semilep/output_${SAMPLE}_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${SAMPLE}_${YEAR}.parquet --dataset ${SAMPLE}_${YEAR} --ntuples $EOS_DIR/non_semilep_ttHbb/${SAMPLE}_${YEAR}/baseline --cat ${CAT}
#    done
    # ttbar semilepton, dilepton, 4FS and tthnon2b are split in subsamples ttLF, ttB, ttC
#    for SAMPLE_TTBAR in "${TTBAR[@]}"; do
#        for SUBSAMPLE in "${SUBSAMPLES[@]}"; do
#            python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_2024.yaml -i $RUN_DIR/output_coffea_non_semilep/output_${SAMPLE_TTBAR}_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${SAMPLE_TTBAR}_${SUBSAMPLE}_${YEAR}.parquet --dataset ${SAMPLE_TTBAR}_${YEAR} --ntuples $EOS_DIR/non_semilep_ttHbb/${SAMPLE_TTBAR}_${YEAR}/${SUBSAMPLE}/baseline --cat ${CAT}
#        done
#    done
#done
