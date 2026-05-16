BASE_DIR="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/fullyhadronic_ttH_export"
PARQUET_DIR=$BASE_DIR/parquet
H5_DIR=$BASE_DIR/h5

mkdir -p $H5_DIR

for YEAR in 2022_preEE 2022_postEE 2023_preBPix 2023_postBPix 2024
do
    FILES=(
        "$PARQUET_DIR"/output_QCD*$YEAR.parquet
        "$PARQUET_DIR"/output_TbarB*$YEAR.parquet
        "$PARQUET_DIR"/output_TbarW*$YEAR.parquet
        "$PARQUET_DIR"/output_TBbar*$YEAR.parquet
        "$PARQUET_DIR"/output_TTH*$YEAR.parquet
        "$PARQUET_DIR"/output_TTW*$YEAR.parquet
        "$PARQUET_DIR"/output_TW*$YEAR.parquet
        "$PARQUET_DIR"/output_Wto2Q*$YEAR.parquet
        "$PARQUET_DIR"/output_Zto2Q*$YEAR.parquet
        "$PARQUET_DIR"/output_TTBBto4Q_ttB*$YEAR.parquet
        "$PARQUET_DIR"/output_TTBBtoLNu2Q_4FS_ttB*$YEAR.parquet
        "$PARQUET_DIR"/output_TTto4Q_ttC*$YEAR.parquet
        "$PARQUET_DIR"/output_TTtoLNu2Q_ttC*$YEAR.parquet
        "$PARQUET_DIR"/output_TTto4Q_ttLF*$YEAR.parquet
        "$PARQUET_DIR"/output_TTtoLNu2Q_ttLF*$YEAR.parquet
    )

    #printf '%s\n' "${FILES[@]}"

    python -m tthbb_spanet.scripts.dataset.parquet_to_h5 --cfg parameters/h5_params/ttHbb_fully_hadronic/features_h5_classification.yaml -i "${FILES[@]}" -o $H5_DIR/output_$YEAR.h5
done
