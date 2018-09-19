#!/usr/bin/env bash

bucket_idx=$1
num_buckets=$2
exp_name=$3
graph=$4

echo "Launching python run-exps-for-fig-4.py --bucket_idx=$bucket_idx --exp-name=$exp_name --graph=$graph"
python -u run_exps_for_fig_4.py --bucket-idx=$bucket_idx --num-buckets=$num_buckets --exp-name=$exp_name --graph=$graph
