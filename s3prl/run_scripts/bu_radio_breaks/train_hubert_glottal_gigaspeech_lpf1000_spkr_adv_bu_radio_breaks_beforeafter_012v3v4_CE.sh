#!/bin/bash
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
ckpt_path=/gscratch/tial/kpever/workspace/prosodyvec/exps/gigaspeech_hubert_glottal_lpf1000_spkr_adv_wt1en1_lr_5em5_train_contd_tmp/checkpoints/checkpoint_best.pt
upstream=prosodyvec_custom
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/bu_radio_breaks/config_prosodyvec_beforeafter_012v3v4_CEloss.yaml
downstream=bu_radio_breaks
output_name=hubert_glottal_gigaspeech_lpf1000_spkr_adv_bu_radio_breaks_beforeafter_012v3v4_CE

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --upstream_ckpt $ckpt_path \
    --expname $output_name