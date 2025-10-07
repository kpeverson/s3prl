#!/bin/bash
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
ckpt_path=/gscratch/tial/kpever/workspace/pretrained_hubert/hubert_base_ls960.pt
upstream=hubert_base
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/bu_radio_tones/config_hubert.yaml
downstream=bu_radio_tones
output_name=hubert_base_bu_radio_tones

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --expname $output_name