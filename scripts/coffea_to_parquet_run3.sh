#!/bin/bash

BASE_DIR="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/fullyhadronic_ttH_export"

for dataset_path in ${BASE_DIR}/*; do
    dataset=$(basename "$dataset_path")

    # Skip folders that are not datasets
    if [[ $dataset == "backup" || $dataset == "parquet" || $dataset == "h5" || $dataset == "v1_MET" || $dataset == "v2_PuppiMET" ]]; then
        continue
    fi

    # Skip ttbar datasets
    if [[ $dataset == TTto* || $dataset == TTBB* ]]; then
        echo "Skipping ttbar dataset: $dataset"
        continue
    fi

    if [[ $dataset == TTH_Hto2B* ]]; then
        sample_type="sig"
    else
        sample_type="bkg"
    fi

    if [[ $dataset == *2022* || $dataset == *2023* ]]; then
        cfg="parameters/ttHbb_fully_hadronic/features_tthbb_FH_${sample_type}_NanoAODv12.yaml"
    elif [[ $dataset == *2024* ]]; then
        cfg="parameters/ttHbb_fully_hadronic/features_tthbb_FH_${sample_type}_NanoAODv15.yaml"
    else
        echo "Error: cannot determine era for dataset: $dataset" >&2
        continue
    fi

    echo "Processing dataset: $dataset with config: $cfg"

    output_dir="$BASE_DIR/parquet/output_${dataset}.parquet"

    # If the file output_dir already exists, skip processing
    if [[ -f "$output_dir" ]]; then
        echo "Output file already exists, skipping dataset: $dataset"
        continue
    fi

    python -m tthbb_spanet.scripts.dataset.coffea_to_parquet \
        -i /eos/user/m/mmarcheg/ttHbb/run3/fullyhadronic/ntuples/multiple_wp_LMT_run3/output_all.coffea \
        --cfg "$cfg" \
        -o "$output_dir" \
        --ntuples "${BASE_DIR}/${dataset}/baseline/nominal/" \
        --dataset "$dataset" \
        --cat baseline
done

# Loop over ttbar datasets and years
TTBAR_SAMPLES=(TTBBto4Q TTBBtoLNu2Q_4FS TTto4Q TTtoLNu2Q)
YEARS=(2022_preEE 2022_postEE 2023_preBPix 2023_postBPix "2024")

for sample in "${TTBAR_SAMPLES[@]}"; do
    for year in "${YEARS[@]}"; do
        dataset="${sample}_${year}"
        dataset_path="${BASE_DIR}/${dataset}"

        # Choose parameter file based on year
        if [[ $year == 2022* || $year == 2023* ]]; then
            cfg="parameters/ttHbb_fully_hadronic/features_tthbb_FH_bkg_NanoAODv12.yaml"
        elif [[ $year == "2024" ]]; then
            cfg="parameters/ttHbb_fully_hadronic/features_tthbb_FH_bkg_NanoAODv15.yaml"
        else
            echo "Error: cannot determine era for dataset: $dataset" >&2
            continue
        fi

        # Loop over subsamples (subfolders): ttB, ttC, ttLF
        for subsample in ttB ttC ttLF; do
            if [[ -d "${dataset_path}/${subsample}" ]]; then
                echo "Processing subsample: ${dataset}/${subsample}"
                output_dir="${BASE_DIR}/parquet/output_${sample}_${subsample}_${year}.parquet"

                # If the file output_dir already exists, skip processing
                if [[ -f "$output_dir" ]]; then
                    echo "Output file already exists, skipping subsample: ${dataset}/${subsample}"
                    continue
                fi

                python -m tthbb_spanet.scripts.dataset.coffea_to_parquet \
                    -i /eos/user/m/mmarcheg/ttHbb/run3/fullyhadronic/ntuples/multiple_wp_LMT_run3/output_all.coffea \
                    --cfg "$cfg" \
                    -o "$output_dir" \
                    --ntuples "${BASE_DIR}/${dataset}/${subsample}/baseline/nominal/" \
                    --dataset "$dataset" \
                    --cat baseline
            else
                echo "Subsample folder does not exist: ${dataset_path}/${subsample}, skipping."
            fi
        done
    done
done
