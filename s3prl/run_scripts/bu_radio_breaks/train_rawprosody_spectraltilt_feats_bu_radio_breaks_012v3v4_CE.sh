#!/bin/bash
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
upstream=prosody_dummy_feats
# upstream=mel
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/bu_radio_breaks/config_rawprosody_012v3v4_CEloss.yaml
downstream=bu_radio_breaks
output_name=rawprosody_spectraltilt_feats_bu_radio_breaks_012v3v4_CE

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --expname $output_name