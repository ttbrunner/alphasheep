#!/bin/bash

#SBATCH -c 4
#SBATCH --mem 16GB
#SBATCH --gres=gpu:1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 srun singularity exec --nv singularity/sng-alphasau.img python3 -u train_rl_agent.py --config="$0"
