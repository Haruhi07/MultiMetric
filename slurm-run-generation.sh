#!/bin/bash

#SBATCH --job-name=gen-summac
#SBATCH --output=gen-summac.txt
#SBATCH --gpus=1
#SBATCH --time=5:00:00


cd /home/b5e/yuxuan.b5e/SelfRL

METHOD=summac STRATEGY=beam6-1+greedy SPLIT=test singularity exec --nv old_torch.sif ./run-generation.sh
