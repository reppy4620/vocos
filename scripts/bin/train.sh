#!/bin/bash

data_dir=../data
download_dir=../downloads

out_dir=../out
mkdir -p $out_dir

python ../src/train.py \
    --train_file $data_dir/label/train.txt \
    --valid_file $data_dir/label/valid.txt \
    --wav_dir    $data_dir/wav24k \
    --lab_dir    $download_dir/jsut-label/labels/basic5000 \
    --out_dir    $out_dir
