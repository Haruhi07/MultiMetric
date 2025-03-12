#!/bin/bash

#SBATCH --job-name=summac-b6-1+b6-2
#SBATCH --output=xsum-summac-beam6-1+beam6-2.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00


cd /home/b5e/yuxuan.b5e/SelfRL
DATASET=beam6-1+beam6-2 METRIC=summac singularity exec --nv old_torch.sif ./run-rl.sh
