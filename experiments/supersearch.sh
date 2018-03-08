#!/usr/bin/env bash

# Paramaterize with the models and datasets you would like to search -- no ',' between items
declare -a models=("cgn")
declare -a datasets=("nslr-syn")
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        export dataset=$dataset
        export model=$model
        sbatch --gres=gpu -x mila00 ./search.sh -m=$model -d=$dataset
    done
done
