#!/bin/bash -eu

download_dir=../downloads
data_dir=../data
mkdir -p $download_dir $data_dir

if [ ! -d $download_dir/jsut_ver1.1 ]; then
    echo "Download JSUT corpus"
    cd $download_dir
    curl -LO http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
    unzip -o jsut_ver1.1.zip
    cd -
fi

if [ ! -d $data_dir/wav24k ]; then
    echo "Resample BASIC5000"
    mkdir $data_dir/wav24k
    for wav_path in $download_dir/jsut_ver1.1/basic5000/wav/*.wav;
    do
        fname=$(basename $wav_path)
        echo $fname
        sox $wav_path -r 24000 $data_dir/wav24k/$fname
    done
fi

if [ ! -d $data_dir/label ]; then
    echo "Preprocess label"
    label_dir=$data_dir/label
    mkdir -p $label_dir
    for wav_path in $data_dir/wav24k/*.wav;
    do
        bname=$(basename $wav_path .wav)
        echo $bname >> $label_dir/all.txt
    done
    tail -n 4900 $label_dir/all.txt > $label_dir/train.txt
    head -n  100 $label_dir/all.txt > $label_dir/valid.txt
fi
