#!/usr/bin/env bash

# Sometimes useful on the MILA cluster
# export HOME=`getent passwd $USER | cut -d':' -f6`
# source activate pytorch0.2

# EXAMPLES OF HOW TO CALL THIS FILE, note that you should run this from the root of the project
# ./experiments/launch_scripts/launch_all_fig_4_jobs.sh 1 experiments/results/genemania genemania
# ./experiments/launch_scripts/launch_all_fig_4_jobs.sh 1 experiments/results/regnet regnet
# Pancan => Genemania
# Kegg => Regnet
# Get the inputs
num_buckets=$1
exp_name=$2
graph=$3
cuda=$4

for bucket_idx in $(seq 1 $num_buckets)
do
    #sbatch --output $exp_dir"/slurm-%j.out" -x mila00 ./experiments/launch_scripts/launch_one_fig_4_job.sh $bucket_idx $num_buckets $exp_name $graph
    ./experiments/launch_scripts/launch_one_fig_4_job.sh $bucket_idx $num_buckets $exp_name $graph $cuda
done
