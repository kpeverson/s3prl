#!/bin/bash

# This script is used to train the MQTTS model on the VoxCeleb1 dataset.
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
ckpt_path=/gscratch/tial/kpever/workspace/prosodyvec/exps/gigaspeech_glottal_lpf1000_normalized_rawprosody_targets_spkr_stopgrad_wt1en1_lr_5em5_train_tmp/checkpoints/checkpoint_best.pt
upstream=prosodyvec_custom
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/voxceleb1/config_prosodyvec_48GB_w_glottal.yaml
downstream=voxceleb1
# feature_selection=features
output_name=prosodyvec_gigaspeech_lpf1000_spkr_stopgrad_voxceleb1

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --upstream_ckpt $ckpt_path \
    --expname $output_name
