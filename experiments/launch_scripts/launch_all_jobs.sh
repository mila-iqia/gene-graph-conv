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
exp=$1
num_buckets=$2
exp_name=$3
graph=$4

for bucket in $(seq 1 $num_buckets)
do
    #sbatch --output $exp_dir"/slurm-%j.out" -x mila00 ./experiments/launch_scripts/launch_one_job.sh $exp $bucket $exp_name $graph
    ./experiments/launch_scripts/launch_one_job.sh $exp $bucket $exp_name $graph
done
