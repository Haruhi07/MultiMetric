#!/bin/bash

#SBATCH --job-name=gen-llama-rlhf-ppo
#SBATCH --output=gen-llama-rlhf-ppo.out
#SBATCH --gpus=1
#SBATCH --time=10:00:00


cd /home/b5e/yuxuan.b5e/SelfRL

./run-llama_generate_with_pipeline.sh
