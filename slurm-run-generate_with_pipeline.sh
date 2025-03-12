#!/bin/bash

#SBATCH --job-name=gen-qwen-b2
#SBATCH --output=gen-qwen-b2.txt
#SBATCH --gpus=1
#SBATCH --time=10:30:00


cd /home/b5e/yuxuan.b5e/SelfRL

STRATEGY=beam6 SPLIT=test NUM_CAND=2 singularity exec --nv new_torch.sif ./run-generate_with_pipeline.sh
