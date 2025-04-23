#!/bin/sh

export DATASET_PATH=completions/xsum/qwen-r1-7b/beam6-1+greedy

python merge_dataset.py \
    --score_dataset_path1 $DATASET_PATH/sbert \
    --score_dataset_path2 $DATASET_PATH/summac \
    --output_path $DATASET_PATH/sbert+summac \
    --n_sample 1000
