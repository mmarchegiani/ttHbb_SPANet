#!/bin/bash

BASE_DIR="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/fullyhadronic_ttH_export-sena"

for dataset_path in ${BASE_DIR}/*; do
    dataset=$(basename "$dataset_path")

    # Only process QCD datasets
    if [[ $dataset != QCD* ]]; then
        echo "Skipping non-QCD dataset: $dataset"
        continue
    fi

    output_dir="./output_${dataset}.parquet"

    echo "Processing dataset: $dataset"

    python -m tthbb_spanet.scripts.dataset.coffea_to_parquet \
        -i /afs/cern.ch/user/s/sedurgut/eos/ttHbb/ttHbb-fully-hadronic/trial_ntuples-0210-all-2/output_all.coffea \
        --cfg parameters/features_tthbb_FH_QCD.yaml \
        -o "$output_dir" \
        --ntuples "${BASE_DIR}/${dataset}/baseline/nominal/" \
        --dataset "$dataset" \
        --cat baseline

done
