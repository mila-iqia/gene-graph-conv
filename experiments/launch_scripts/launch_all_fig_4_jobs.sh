#!/usr/bin/env bash

# Sometimes useful on the MILA cluster
# export HOME=`getent passwd $USER | cut -d':' -f6`
# source activate pytorch0.2

# EXAMPLES OF HOW TO CALL THIS FILE, note that you should run this from the root of the project
# ./experiments/launch_scripts/launch_all_jobs.sh fig-4 1 experiments/results/genemania_first genemania
# ./experiments/launch_scripts/launch_all_jobs.sh fig-4 10 experiments/results/regnet_100 regnet

# Pancan => Genemania
# Kegg => Regnet
# Get the inputs
num_buckets=$1
exp_name=$2
graph=$3

for bucket_idx in $(seq 1 $num_buckets)
do
    #sbatch --output $exp_dir"/slurm-%j.out" -x mila00 ./experiments/launch_scripts/launch_one_fig_4_job.sh $bucket_idx $num_buckets $exp_name $graph
    ./experiments/launch_scripts/launch_one_fig_4_job.sh $bucket_idx $num_buckets $exp_name $graph
done
