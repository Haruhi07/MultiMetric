#!/bin/bash

#SBATCH --job-name=gen-llama-mpo
#SBATCH --output=gen-xsum-mpo-beam6-1+greedy.txt
#SBATCH --gpus=1
#SBATCH --time=4:00:00


cd /home/b5e/yuxuan.b5e/SelfRL

METHOD=mpo STRATEGY=beam6-1+greedy singularity exec --nv old_torch.sif ./run-predict_with_pipeline.sh
