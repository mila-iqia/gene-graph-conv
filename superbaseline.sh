#!/usr/bin/env bash
declare -a models=("mlp" "slr" "cgn")
declare -a datasets=("percolate-plus")
declare -a param_values=(50 100)
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
      for value in "${param_values[@]}"
      do
        export dataset=$dataset
        export model=$model
        export param=$value
        ./baseline.sh -m=$model -d=$dataset -p=$value
      done
    done
done
