python scripts/dataset/parquet_to_h5.py --cfg parameters/features_spanet.yaml -i /eos/user/m/mmarcheg/ttHbb/training_datasets/2018/output_ttHTobb_ttToSemiLep_2018.parquet /eos/user/m/mmarcheg/ttHbb/training_datasets/2017/output_ttHTobb_ttToSemiLep_2017.parquet /eos/user/m/mmarcheg/ttHbb/training_datasets/2016_PostVFP/output_ttHTobb_ttToSemiLep_2016_PostVFP.parquet /eos/user/m/mmarcheg/ttHbb/training_datasets/2016_PreVFP/output_ttHTobb_ttToSemiLep_2016_PreVFP.parquet -o /eos/user/m/mmarcheg/ttHbb/training_datasets/full_Run2/with_electron_flag/output_ttHTobb_ttToSemiLep_full_Run2_with_electron_flag.h5
python scripts/dataset/parquet_to_h5.py --cfg parameters/features_spanet.yaml -i /eos/user/m/mmarcheg/ttHbb/training_datasets/2018/output_ttHTobb_ttToSemiLep_2018.parquet /eos/user/m/mmarcheg/ttHbb/training_datasets/2017/output_ttHTobb_ttToSemiLep_2017.parquet /eos/user/m/mmarcheg/ttHbb/training_datasets/2016_PostVFP/output_ttHTobb_ttToSemiLep_2016_PostVFP.parquet /eos/user/m/mmarcheg/ttHbb/training_datasets/2016_PreVFP/output_ttHTobb_ttToSemiLep_2016_PreVFP.parquet -o /eos/user/m/mmarcheg/ttHbb/training_datasets/full_Run2/with_electron_flag/output_ttHTobb_ttToSemiLep_full_Run2_with_electron_flag_fullymatched.h5 --fully_matched