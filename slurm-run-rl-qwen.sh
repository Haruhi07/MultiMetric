#!/bin/bash

#SBATCH --job-name=q-x-sbert-b6-1+gdy
#SBATCH --output=qwen-xsum-sbert-beam6-1+greedy.out
#SBATCH --gpus=1
#SBATCH --nodelist=nid001036
#SBATCH --time=23:59:00


cd /home/b5e/yuxuan.b5e/SelfRL
BNB_CUDA_VERSION=122 ORIGIN=xsum DATASET=beam6-1+greedy METRIC=sbert SCORE_DATASET=True ./run-rl-qwen.sh
