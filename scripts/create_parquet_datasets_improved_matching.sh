# Nested loops over year and datasets ttHTobb and ttHTobb_ttToSemiLep to create parquet datasets with improved matching
for year in 2016_PreVFP 2016_PostVFP 2017 2018
do
    for dataset in ttHTobb ttHTobb_ttToSemiLep
    do
        python scripts/dataset/coffea_to_parquet.py --cfg parameters/features_improved_matching.yaml -i /eos/user/m/mmarcheg/ttHbb/training_datasets/sig_bkg_ntuples_ttHTobb_ttToSemiLep_improved_matching/output_all.coffea -o /eos/user/m/mmarcheg/ttHbb/training_datasets/sig_bkg_ntuples_ttHTobb_ttToSemiLep_improved_matching/parquet/output_${dataset}_${year}.parquet --ntuples /eos/user/m/mmarcheg/ttHbb/training_datasets/sig_bkg_ntuples_ttHTobb_ttToSemiLep_improved_matching/ntuples/output_columns_parton_matching/parton_matching_20_06_24/${dataset}_${year}/semilep_LHE
    done
done
