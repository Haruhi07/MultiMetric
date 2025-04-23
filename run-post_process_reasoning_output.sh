#!/bin/sh

python post_process_reasoning_output.py \
    --csv_path completions/qwen-r1-7b-xsum/test-beam6-1.csv \
    --save_path completions/qwen-r1-7b-xsum/processed/test-beam6-1.csv