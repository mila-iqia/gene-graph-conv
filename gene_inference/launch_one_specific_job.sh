#!/usr/bin/env bash


# Launch one thing. Depends on local stuff. change this file if needed.
#source activate venv/bin
bucket=$1
exp_dir=$2
graph_path=$3
first_degree=$4
second_degree=$5
#python -W ignore ../../../baseline.py --log=silent --cuda --baseline-model=$model --baseline-dataset=$dataset

echo "Launching python run_experiments.py --bucket=$bucket --exp-dir=$exp_dir --graph-path=$graph_path"
python -u run_experiments.py --bucket=$bucket --exp-dir=$exp_dir --graph-path=$graph_path --first-degree=$first_degree
