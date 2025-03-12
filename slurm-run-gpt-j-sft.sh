#!/bin/bash

#SBATCH --job-name=sft-gpt-j
#SBATCH --output=sft-gpt-j.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00


cd /home/b5e/yuxuan.b5e/SelfRL
singularity exec --nv old_torch.sif ./run-gpt-j-sft.sh
