#!/bin/bash
source /gscratch/tial/kpever/miniconda3/bin/activate superb

s3prl_dir=/gscratch/tial/kpever/workspace/s3prl/s3prl
upstream=hubert_base
downstream_config_path=/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/bu_radio_breaks/config_hubert_0123v4_ContinuumCEloss.yaml
downstream=bu_radio_breaks
output_name=hubert_base_bu_radio_breaks_0123v4_continuumCE

python $s3prl_dir/run_downstream.py \
    --mode train \
    --downstream $downstream \
    --config $downstream_config_path \
    --upstream $upstream \
    --expname $output_name