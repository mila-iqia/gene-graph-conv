#!/usr/bin/env bash
#declare -a models=("lr_no_l1" "lr_with_l1" "slr_no_l1" "slr_with_l1" "mlp" "cgn_pool" "cgn_no_pool")
declare -a models=("cgn_pool" "cgn_no_pool")
declare -a datasets=("percolate-plus")
#declare -a param_values=(0 10 20 30 40 50 60 70 80 90 100)
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        export dataset=$dataset
        export model=$model
        sbatch --gres=gpu -x mila00 ./baseline.sh -m=$model -d=$dataset #-p=$value
       #./baseline.sh -m=$model -d=$dataset #-p=$value
    done
done
