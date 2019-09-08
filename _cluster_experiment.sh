#!/bin/bash
# Starts a single train job and 3 eval jobs to run on the cluster.
# Since we are a lazy bunch, we simply fork lots of processes instead of designing a a more efficient implementation :)

sbatch --qos=lowprio _cluster_runner/slurm_train_alphasheep.sh "$1"
sbatch --qos=lowprio _cluster_runner/slurm_eval_alphasheep.sh "$1"
sbatch --qos=lowprio _cluster_runner/slurm_eval_alphasheep.sh "$1"
sbatch --qos=lowprio _cluster_runner/slurm_eval_alphasheep.sh "$1"
