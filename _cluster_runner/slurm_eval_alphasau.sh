#!/bin/bash

#SBATCH -c 2
#SBATCH --mem 10GB
# No GPU here, it's probably not worth it

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 srun singularity exec --nv _cluster_runner/sng-alphasau.img python3 -u eval_rl_agent.py --config="$1" --loop
