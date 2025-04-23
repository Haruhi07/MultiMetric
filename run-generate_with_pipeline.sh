#!/bin/sh

export LD_LIBRARY_PATH=/home/b5e/yuxuan.b5e/.local/lib/python3.12/site-packages/bitsandbytes/:$LD_LIBRARY_PATH
export PATH=/home/b5e/yuxuan.b5e/.local/bin:$PATH

python3 generate_with_pipeline.py \
    --save_path completions/qwen-r1-7b-$ORIGIN-dpo/$METRIC-$STRATEGY-$PART.csv \
    --decoding_strategy beam6 \
    --num_return_sequences 1 \
    --model_name_or_path qwen-r1-7b-$ORIGIN-dpo-$STRATEGY/$METRIC/final_merged_checkpoint \
    --tokenizer_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name_or_path $DATASET \
    --split test \
    --max_doc_length 1000 \
    --max_new_tokens 1000 \
    --batch_size 8
