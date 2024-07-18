years=("2016_PreVFP" "2016_PostVFP" "2017")

for year in ${years[@]}; do
    python scripts/dataset/coffea_to_parquet.py --cfg parameters/features.yaml -i /eos/user/m/mmarcheg/ttHbb/training_datasets/${year}/output_TTbbSemiLeptonic_4f_${year}.coffea -o /eos/user/m/mmarcheg/ttHbb/training_datasets/${year}/output_TTbbSemiLeptonic_${year}.parquet
done
