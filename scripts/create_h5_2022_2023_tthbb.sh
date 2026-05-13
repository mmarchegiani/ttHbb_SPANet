#!/bin/bash

BASE_DIR="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/fullyhadronic_ttH_export-sena"

for dataset_path in ${BASE_DIR}/*; do
    dataset=$(basename "$dataset_path")

    # Only process QCD datasets
    #if [[ $dataset != QCD* ]]; then
    #    echo "Skipping non-QCD dataset: $dataset"
    #    continue
    #fi
    #only process non-QCD datasets
    if [[ $dataset == QCD* ]]; then
        echo "Skipping QCD dataset: $dataset"
        continue
    fi

    output_dir="./h5/output_${dataset}.h5"

    echo "Processing dataset: $dataset"

    python -m tthbb_spanet.scripts.dataset.parquet_to_h5 \
        -i ./output_${dataset}.parquet \
        --cfg parameters/features_tthbb_FH_QCD.yaml \
        -o "$output_dir" 
done
