#!/usr/bin/env bash

exp=$1
bucket=$2
exp_name=$3
graph=$4

echo "Launching python run-exps-for-fig-4.py --bucket=$bucket --exp-name=$exp_name --graph=$graph"
if test $exp="fig-4";
  then python -u run-exps-for-fig-4.py --bucket=$bucket --exp-name=$exp_name --graph=$graph;
elif test $exp="fig-5";
  then python -u run-exps-for-fig-5.py --bucket=$bucket --exp-name=$exp_name --graph=$graph;
else
  echo "enter a valid figure id";
fi
