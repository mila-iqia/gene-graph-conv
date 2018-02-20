#!/usr/bin/env bash
declare -a models=("cgn_pool")
declare -a datasets=("percolate-plus")
if [ $# -eq 0 ]
  then
    exit
fi
for i in "$@"
do
case $i in
    -e=*|--experiment=*)
    experiment="${i#*=}"
    ;;
esac
done

DATE=$(date +%F%k%M%S)
declare -a exp_dir="data/$experiment/"
declare -a data_dir=$DATE
declare -a dir=$exp_dir$data_dir
mkdir $dir
cp baseline.sh $dir"/baseline.sh"
cp $exp_dir"settings.json" $dir"/settings.json"
cd $dir

for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        export dataset=$dataset
        export model=$model
        sbatch --gres=gpu:titanx -x mila00 ./baseline.sh -m=$model -d=$dataset
    done
done
