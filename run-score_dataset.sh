#!/bin/sh

export RAWDATA_PATH=completions/qwen-r1-7b-xsum
export DATASET_PATH=completions/xsum/qwen-r1-7b

python score_dataset.py \
    --raw_dataset_path1 $RAWDATA_PATH/train-beam6-1.csv \
    --raw_dataset_path2 $RAWDATA_PATH/train-$LOSE.csv \
    --output_path $DATASET_PATH/beam6-1+$LOSE/summac/train

python score_dataset.py \
    --raw_dataset_path1 $RAWDATA_PATH/test-beam6-1.csv \
    --raw_dataset_path2 $RAWDATA_PATH/test-$LOSE.csv \
    --output_path $DATASET_PATH/beam6-1+$LOSE/summac/test