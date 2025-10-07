#!/bin/bash
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
upstream=prosody_dummy_feats
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/bu_radio_tones/config_rawprosody.yaml
downstream=bu_radio_tones
output_name=rawprosody_spectraltilt_feats_bu_radio_tones

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --expname $output_name