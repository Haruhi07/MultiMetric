#!/bin/sh

python3 sampling_with_different_strategy.py \
    --save_path completions/gpt-j-6b-xsum/temp1-5-$NUM_CAND.csv \
    --decoding_strategy temp1.5 \
    --num_return_sequences $NUM_CAND \
    --model_name_or_path gpt-j-6b-xsum-sft/final_merged_checkpoint \
    --tokenizer_name_or_path EleutherAI/gpt-j-6b \
    --dataset_name_or_path EdinburghNLP/xsum \
    --split test \
    --max_doc_length 2000 \
    --max_new_tokens 50 \
    --batch_size 1
