#!/usr/bin/env bash
declare -a models=("mlp" "slr" "cgn")
declare -a datasets=("nslr-syn")
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        export dataset=$dataset
        export model=$model
        sbatch --gres=gpu -x mila00 ./baseline.sh -m=$model -d=$dataset
    done
done
