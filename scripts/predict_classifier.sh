MODEL=/eos/user/m/mmarcheg/ttHbb/models/classifier_2017/spanet_output/version_1
PREDICTION=/eos/user/m/mmarcheg/ttHbb/predictions/classifier_2017/spanet_output/version_1/predictions_ttHTobb_ttToSemiLep_ttToSemiLeptonic_2017_inclusive_validation_733186.h5
VALIDATION=/eos/user/m/mmarcheg/ttHbb/training_datasets/classifier/with_signal_flag/ttHTobb_ttToSemiLep_TTToSemiLeptonic_2017_test_733186.h5

# Create output directory if it does not exist
mkdir -p $(dirname $PREDICTION)
python -m spanet.predict $MODEL $PREDICTION -tf $VALIDATION --gpu
