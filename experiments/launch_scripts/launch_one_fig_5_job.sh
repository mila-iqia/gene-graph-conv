#!/usr/bin/env bash

gene=$1
graph_path=$2
samples_path=$3
trials=$4
cuda=$5
echo $graph_path
for i in "$@"
do
case $i in
    --gene=*)
    gene="${i#*=}"
    ;;
    --graph-path=*)
    graph_path="${i#*=}"
    ;;
    --samples-path=*)
    samples_path="${i#*=}"
    ;;
    --trials=*)
    trials="${i#*=}"
    ;;
    --cuda=*)
    cuda="${i#=}"
    ;;
esac
done

job_to_run="python -u run_exps_for_fig_5.py"
job_to_run=$job_to_run" --gene="$gene
if [ $cuda ]
then
    job_to_run=$job_to_run" --cuda ";
fi
if [ $graph_path ]
then
    job_to_run=$job_to_run" --graph-path="$graph_path;
fi
if [ $trials ]
then
    job_to_run=$job_to_run" --trials="$trials;
fi
if [ $samples_path ]
then
    job_to_run=$job_to_run" --samples-path="$samples_path;
fi
$job_to_run
