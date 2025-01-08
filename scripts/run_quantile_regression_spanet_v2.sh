PARQUET_FOLDER=/eos/user/m/mmarcheg/ttHbb/training_datasets/spanet_v2/parquet/spanet_inference

python scripts/quantile_regression.py -i ${PARQUET_FOLDER}/2016_PreVFP/output_ttHTobb_2016_PreVFP.parquet ${PARQUET_FOLDER}/2016_PostVFP/output_ttHTobb_2016_PostVFP.parquet ${PARQUET_FOLDER}/2017/output_ttHTobb_2017.parquet ${PARQUET_FOLDER}/2018/output_ttHTobb_2018.parquet -o output_quantile_regression_spanet_v2 -n 1000000
