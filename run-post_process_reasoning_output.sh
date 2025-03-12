#!/bin/sh

python post_process_reasoning_output.py \
    --csv_path completions/qwen-r1-7b-tldr/test-temp1.5-1.csv \
    --save_path completions/qwen-r1-7b-tldr/processed/test-temp1.5-1.csv