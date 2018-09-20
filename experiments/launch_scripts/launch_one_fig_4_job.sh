#!/usr/bin/env bash

bucket_idx=$1
num_buckets=$2
exp_name=$3
graph=$4
cuda=$5
job_to_run="python -u run_exps_for_fig_4.py --bucket-idx=$bucket_idx --num-buckets=$num_buckets --exp-name=$exp_name --graph=$graph"
if [ $cuda ]
then
    job_to_run=$job_to_run" --cuda ";
fi
echo $job_to_run
$job_to_run
