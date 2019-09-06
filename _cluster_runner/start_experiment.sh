#!/bin/bash
# Starts a single train job and 4 eval jobs to run on the cluster.
# Since we are a lazy bunch, we simply fork lots of processes instead of designing a a more efficient implementation :)

sbatch --qos=lowprio singularity/slurm_train_alphasau.sh "$0"
sbatch --qos=lowprio singularity/slurm_eval_alphasau.sh "$0"
sbatch --qos=lowprio singularity/slurm_eval_alphasau.sh "$0"
sbatch --qos=lowprio singularity/slurm_eval_alphasau.sh "$0"
sbatch --qos=lowprio singularity/slurm_eval_alphasau.sh "$0"
