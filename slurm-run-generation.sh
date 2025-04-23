#!/bin/bash

#SBATCH --job-name=gen-gptj-1-5-2
#SBATCH --output=gen-summac.txt
#SBATCH --gpus=1
#SBATCH --time=5:00:00


cd /home/b5e/yuxuan.b5e/SelfRL

NUM_CAND=2 singularity exec --nv old_torch.sif ./run-generation.sh
