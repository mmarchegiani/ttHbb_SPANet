#!/bin/bash

RUNS=(
  /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_full/classifier_btag_full/stitched_v4_v7/
  /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_T/classifier_btag_T/version_1
  /home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_TM/classifier_btag_TM/version_4
/home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_TML/version_0
/home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/classifier_btag_TML_QvG/classifier_btag_TML_QvG/version_0
)

LABELS=(
  "btag_full"
  "btag_T"
  "btag_TM"
  "btag_TML"
  "btag_TML_QvG"
)

TITLE="Signal classification accuracy"
MAX_EPOCH=25
OUT="/home/export/sdurgut/scratch/ttHbb_SPANet/training_metrics/common_plots/compare_signal_accuracy_fully_hadronic.png"
CMS_SUBTITLE="Internal work"
python scripts/ttHbb_fully_hadronic/compare_signal_accuracy_runs.py \
  --runs "${RUNS[@]}" \
  --labels "${LABELS[@]}" \
  --title "${TITLE}" \
  --cms-subtitle "${CMS_SUBTITLE}" \
  --max-epoch "${MAX_EPOCH}" \
  --out "${OUT}"