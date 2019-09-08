#!/bin/bash

#SBATCH -c 4
#SBATCH --mem 12GB
#SBATCH --gres=gpu:1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 srun singularity exec --nv _cluster_runner/sng-alphasheep.img python3 -u train_rl_agent.py --config="$1"
