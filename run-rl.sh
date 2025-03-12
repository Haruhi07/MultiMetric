#!/bin/sh

export PATH=/home/b5e/yuxuan.b5e/.local/bin:$PATH

python3 rl.py \
    --score_dataset \
    --output_dir  llama-3b-xsum-dpo-$DATASET/$METRIC \
    --dataset_name completions/xsum/llama-3b-instruct/$DATASET/$METRIC \
    --run_name llama-3b-xsum-dpo-$DATASET-$METRIC \
    --logging_dir llama-3b-xsum-dpo-$DATASET/log/$METRIC \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --learning_rate 1e-4 \
    --warmup_steps 150 \
    --num_train_epochs 1 \
    --max_steps 1000 \
    --save_total_limit 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps 200 \
    --no_remove_unused_columns \
    --report_to tensorboard \
    --max_prompt_length 2000 \
    --max_length 2048 \
    --load_in_4bit True \
    --lora_alpha 32 \
    --lora_r 16 \
    --lora_dropout 0.05 \
    --beta 0.5
