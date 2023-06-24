#!/bin/bash

out_dir=../out/generate

python ../src/synthesize.py \
    --label_file ../data/label/valid.txt \
    --wav_dir    ../data/wav24k \
    --ckpt_path  ../out/ckpt/last.ckpt \
    --out_dir    $out_dir
