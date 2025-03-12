#!/bin/sh

python3 sampling_with_different_strategy.py \
    --save_path completions/gpt-j-6b-xsum-dpo/$METHOD-$STRATEGY.csv \
    --decoding_strategy beam6 \
    --num_return_sequences 1 \
    --model_name_or_path gpt-j-6b-xsum-dpo-$STRATEGY/$METHOD/final_merged_checkpoint \
    --tokenizer_name_or_path EleutherAI/gpt-j-6b \
    --dataset_name_or_path EdinburghNLP/xsum \
    --split $SPLIT \
    --max_doc_length 2000 \
    --max_new_tokens 50 \
    --batch_size 4
