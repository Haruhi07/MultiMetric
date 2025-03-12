#!/bin/sh

export RAWDATA_PATH=completions/llama-3b-instruct-tldr
export DATASET_PATH=completions/tldr/llama-3b-instruct

python score_dataset.py \
    --raw_dataset_path1 $RAWDATA_PATH/train-beam6-1.csv \
    --raw_dataset_path2 $RAWDATA_PATH/train-beam6-2.csv \
    --output_path $DATASET_PATH/beam6-1+beam6-2/summac/train

python score_dataset.py \
    --raw_dataset_path1 $RAWDATA_PATH/test-beam6-1.csv \
    --raw_dataset_path2 $RAWDATA_PATH/test-beam6-2.csv \
    --output_path $DATASET_PATH/beam6-1+beam6-2/summac/test