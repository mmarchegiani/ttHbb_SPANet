MODEL=/eos/user/m/mmarcheg/ttHbb/models/multiclassifier_2018_300epochs/spanet_output/version_0
PREDICTION=/eos/user/m/mmarcheg/ttHbb/predictions/multiclassifier_2018_300epochs/spanet_output/version_0/predictions_tthbb_ttbar_with_ttbb_merging_2018_test_919484_epoch55.h5
VALIDATION=/eos/user/m/mmarcheg/ttHbb/training_datasets/classifier/multiclassifier_2018/tthbb_ttbar_with_ttbb_merging_2018_test_919484.h5

# Create output directory if it does not exist
mkdir -p $(dirname $PREDICTION)
python -m spanet.predict $MODEL $PREDICTION -tf $VALIDATION --gpu
