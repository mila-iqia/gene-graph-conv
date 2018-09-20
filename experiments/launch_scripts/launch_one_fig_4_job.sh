#!/usr/bin/env bash

for i in "$@"
do
case $i in
    --bucket-idx=*)
    bucket_idx="${i#*=}"
    ;;
    --num-buckets=*)
    num_buckets="${i#*=}"
    ;;
    --graph-path=*)
    graph_path="${i#*=}"
    ;;
    --samples-path=*)
    samples_path="${i#*=}"
    ;;
    --exp-name=*)
    exp_name="${i#=}"
    ;;
    --cuda=*)
    cuda="${i#=}"
    ;;
esac
done
echo $exp_name
job_to_run="python -u run_exps_for_fig_4.py"
if [ $cuda ]
then
    job_to_run=$job_to_run" --cuda ";
fi
if [ $bucket_idx ]
then
    job_to_run=$job_to_run" --bucket-idx="$bucket_idx;
fi
if [ $num_buckets ]
then
    job_to_run=$job_to_run" --num-buckets="$num_buckets;
fi
if [ $graph_path ]
then
    job_to_run=$job_to_run" --graph-path="$graph_path;
fi
if [ $exp_name ]
then
    job_to_run=$job_to_run" --exp-name="$exp_name;
fi
if [ $samples_path ]
then
    job_to_run=$job_to_run" --samples-path="$samples_path;
fi
echo $job_to_run
$job_to_run
