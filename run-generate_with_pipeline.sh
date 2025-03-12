#!/bin/sh

python3 generate_with_pipeline.py \
    --save_path completions/qwen-r1-7b-xsum/$SPLIT-$STRATEGY-$NUM_CAND.csv \
    --decoding_strategy $STRATEGY \
    --num_return_sequences $NUM_CAND \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tokenizer_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name_or_path EdinburghNLP/xsum \
    --split $SPLIT \
    --max_doc_length 1000 \
    --max_new_tokens 1000 \
    --batch_size 4
