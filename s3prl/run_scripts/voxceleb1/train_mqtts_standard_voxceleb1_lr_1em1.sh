#!/bin/bash

# This script is used to train the MQTTS model on the VoxCeleb1 dataset.
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
ckpt_path=/gscratch/tial/kpever/workspace/mqtts_training/quantizer/checkpoints/mqtts_quantizer_standard/g_00300000
upstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/upstream/mqtts_quantizer/configs/config_standard.json
upstream=mqtts_custom
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/voxceleb1/config_48GB_lr_1em1.yaml
downstream=voxceleb1
feature_selection=features
output_name=mqtts_standard_voxceleb1_lr_1em1

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --upstream_ckpt $ckpt_path \
    --upstream_model_config $upstream_config_path \
    --upstream_feature_selection $feature_selection \
    --expname $output_name
