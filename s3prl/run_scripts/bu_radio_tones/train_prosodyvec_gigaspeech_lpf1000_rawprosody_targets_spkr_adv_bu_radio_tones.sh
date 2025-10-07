#!/bin/bash
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
ckpt_path=/gscratch/tial/kpever/workspace/prosodyvec/exps/old_gigaspeech_targets/gigaspeech_glottal_lpf1000_normalized_rawprosody_targets_spkr_adv_wt1en1_lr_5em5_train_tmp/checkpoints/checkpoint_best.pt
upstream=prosodyvec_custom
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/bu_radio_tones/config_prosodyvec.yaml
downstream=bu_radio_tones
output_name=prosodyvec_gigaspeech_lpf1000_rawprosody_targets_spkr_adv_bu_radio_tones

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --upstream_ckpt $ckpt_path \
    --expname $output_name