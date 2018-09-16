#!/usr/bin/env bash

exp=$1
bucket_idx=$2
num_buckets=$3
exp_name=$4
graph=$5

echo "Launching python run-exps-for-fig-4.py --bucket_idx=$bucket_idx --exp-name=$exp_name --graph=$graph"
if test $exp="fig-4";
  then python -u run-exps-for-fig-4.py --bucket-idx=$bucket_idx --num-buckets=$num_buckets --exp-name=$exp_name --graph=$graph;
elif test $exp="fig-5";
  then python -u run-exps-for-fig-5.py  --bucket-idx=$bucket_idx --num-buckets=$num_buckets --exp-name=$exp_name --graph=$graph;
else
  echo "enter a valid figure id";
fi
