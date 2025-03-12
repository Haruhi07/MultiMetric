#!/bin/sh

export PATH=/home/b5e/yuxuan.b5e/.local/bin:$PATH

python3 gpt-j-sft.py \
    --output_dir  gpt-j-6b-xsum-sft \
    --dataset_name  EdinburghNLP/xsum\
    --run_name gpt-j-6b-xsum-sft \
    --logging_dir gpt-j-6b-xsum-sft/log \
    --model_name_or_path CarperAI/openai_summarize_tldr_sft \
    --learning_rate 5e-4 \
    --max_steps 10000 \
    --warmup_steps 100 \
    --num_train_epochs 3 \
    --save_total_limit 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing False \
    --weight_decay 0.05 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 True \
    --no_remove_unused_columns \
    --report_to tensorboard \
    --max_seq_length 1024 \
    --load_in_4bit True \
    --lora_alpha 32 \
    --lora_r 16 \
    --lora_dropout 0.05
