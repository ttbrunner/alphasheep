#!/bin/bash

#SBATCH -c 4
#SBATCH --mem 16GB
# No GPU here, it's probably not worth it

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 srun singularity exec --nv singularity/sng-alphasau.img python3 -u eval_rl_agent.py alphasau-current.h5 --loop
