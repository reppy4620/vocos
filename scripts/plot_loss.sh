#!/bin/bash

out_dir=../out/plot

python ../src/plot_loss.py \
    --loss_file ../out/logs/version_1/metrics.csv \
    --out_dir $out_dir