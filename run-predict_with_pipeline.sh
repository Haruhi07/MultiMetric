#!/bin/sh

python3 generate_with_pipeline.py \
    --save_path completions/llama-3b-instruct-xsum-dpo/$METHOD-$STRATEGY.csv \
    --decoding_strategy beam6 \
    --num_return_sequences 1 \
    --model_name_or_path  llama-3b-xsum-dpo-$STRATEGY/$METHOD/final_merged_checkpoint\
    --dataset_name_or_path EdinburghNLP/xsum \
    --tokenizer_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --split test \
    --max_doc_length 2000 \
    --max_new_tokens 50 \
    --batch_size 4
