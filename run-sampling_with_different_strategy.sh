#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
export METHOD=mpo
export STRATEGY=beam6-1+greedy
export SPLIT=test

python3 sampling_with_different_strategy.py \
    --save_path completions/gpt-j-6b-xsum/temp1-5-1.csv \
    --decoding_strategy temp1.5 \
    --num_return_sequences 1 \
    --model_name_or_path gpt-j-6b-xsum-sft/final_merged_checkpoint \
    --tokenizer_name_or_path EleutherAI/gpt-j-6b \
    --dataset_name_or_path EdinburghNLP/xsum \
    --split $SPLIT \
    --max_doc_length 2000 \
    --max_new_tokens 50 \
    --batch_size 1
