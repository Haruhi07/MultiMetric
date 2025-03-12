#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export METRIC=sbert+summac_remove_conflicts
export STRATEGY=beam6-1+beam6-2

python rl-bart.py \
    --output_dir bart-rlhf/dpo \
    --dataset_name trl-lib/tldr-preference \
    --run_name bart-tldr-rlhf-dpo \
    --logging_dir bart-rlhf/log/dpo \
    --model_name_or_path bart-tldr \
    --learning_rate 1.0e-6 \
    --num_train_epochs 2 \
    --save_total_limit 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 500 \
    --no_remove_unused_columns \
    --report_to tensorboard \
    --max_prompt_length 512 \
    --max_completion_length 62 
