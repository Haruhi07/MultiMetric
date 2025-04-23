#!/bin/sh

export BNB_CUDA_VERSION=122

python3 llama_generate_with_pipeline.py \
    --save_path completions/llama-3b-instruct-rlhf/ppo.csv \
    --decoding_strategy beam6 \
    --num_return_sequences 1 \
    --model_name_or_path  /home/b5e/yuxuan.b5e/SelfRL/llama-3b-rlhf/ppo/final_merged_checkpoint\
    --dataset_name_or_path trl-lib/tldr \
    --tokenizer_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --split test \
    --max_doc_length 2000 \
    --max_new_tokens 50 \
    --batch_size 4
