#!/bin/bash

#SBATCH --job-name=mpo-b6-1+gdy
#SBATCH --output=tldr-mpo-beam6-1+greedy.out
#SBATCH --gpus=1
#SBATCH --time=18:00:00


cd /home/b5e/yuxuan.b5e/SelfRL
ORIGIN=tldr DATASET=beam6-1+greedy METRIC=mpo SCORE_DATASET=True singularity exec --nv old_torch.sif ./run-rl.sh
