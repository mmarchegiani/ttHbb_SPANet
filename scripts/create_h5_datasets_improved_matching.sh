# Read multiple parquet files and create a joint h5 dataset
basedir="/eos/user/m/mmarcheg/ttHbb/training_datasets/sig_bkg_ntuples_ttHTobb_ttToSemiLep_improved_matching/parquet"
dataset="ttHTobb_ttToSemiLep"
python scripts/dataset/parquet_to_h5.py --cfg parameters/features_spanet.yaml -i $basedir/output_${dataset}_2016_PreVFP.parquet $basedir/output_${dataset}_2016_PostVFP.parquet $basedir/output_${dataset}_2017.parquet $basedir/output_${dataset}_2018.parquet -o /eos/user/m/mmarcheg/ttHbb/training_datasets/sig_bkg_ntuples_ttHTobb_ttToSemiLep_improved_matching/h5/output_${dataset}_full_Run2.h5
