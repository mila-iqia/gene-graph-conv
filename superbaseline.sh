#!/usr/bin/env bash
declare -a models=("lr_no_l1" "lr_with_l1" "slr_with_l1" "mlp" "cgn_pool" "cgn_no_pool" "cgn_dropout" "random")
#declare -a models=("lr_no_l1")
declare -a datasets=("synmin")
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

DATE=$(date +%F%k%M)
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
        sbatch --gres=gpu -x mila00 ./baseline.sh -m=$model -d=$dataset
    done
done
