#!/bin/bash

# This script is used to train the MQTTS model on the VoxCeleb1 dataset.
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
upstream=hubert_base
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/voxceleb1/config_prosodyvec_48GB_w_glottal.yaml
downstream=voxceleb1
# feature_selection=features
output_name=hubert_final_layer_voxceleb1

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --expname $output_name \
    --upstream_layer_selection -1
