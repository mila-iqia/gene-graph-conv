#!/usr/bin/env bash

# Get the inputs
args=$1
trials=$2
exp_dir=$3

for i in $(seq 1 $trials)
do
    exp_args="$args --seed $i"
    echo "launching" $exp_args
    sbatch --output $exp_dir"/slurm-%j.out" --gres=gpu:titanx -x mila00 ./launch_one_specific_job.sh $exp_args
done