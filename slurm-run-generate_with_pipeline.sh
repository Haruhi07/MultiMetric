#!/bin/bash

#SBATCH --job-name=gen2-qwen-xsum-mpo-b6-1+gdy
#SBATCH --output=gen2-qwen-xsum-mpo-b6-1+gdy.txt
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00


cd /home/b5e/yuxuan.b5e/SelfRL
source venv/bin/activate

BNB_CUDA_VERSION=122 ORIGIN=xsum DATASET=EdinburghNLP/xsum METRIC=mpo STRATEGY=beam6-1+greedy PART=2 ./run-generate_with_pipeline.sh
