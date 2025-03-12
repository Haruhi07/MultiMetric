#!/bin/sh

export DATASET_PATH=completions/tldr/llama-3b-instruct/beam6-1+beam6-2

python merge_dataset.py \
    --score_dataset_path1 $DATASET_PATH/sbert \
    --score_dataset_path2 $DATASET_PATH/summac \
    --output_path $DATASET_PATH/sbert+summac_remove_conflicts \
    --remove_conflicts \
    --n_sample 1000
