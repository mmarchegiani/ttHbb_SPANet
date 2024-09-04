for year in 2016_PreVFP 2016_PostVFP 2017 2018
do
    python scripts/quantile_regression.py -i /eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/${year} -o output_quantile_regression/${year} --cfg parameters/features_spanet_quantile_transformer.yaml -n 1000000
done
