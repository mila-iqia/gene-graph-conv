#!/usr/bin/env bash

# Sometimes useful on the MILA cluster
# export HOME=`getent passwd $USER | cut -d':' -f6`
# source activate pytorch0.2

# EXAMPLES OF HOW TO CALL THIS FILE, note that you should run this from the root of the project
# ./experiments/launch_scripts/launch_all_fig_4_jobs.sh --num-buckets=1 --exp-name=genemania
# ./experiments/launch_scripts/launch_all_fig_4_jobs.sh --num-buckets=1 --exp-name=genemania
# on Mila Cluster ./experiments/launch_scripts/launch_all_fig_4_jobs.sh --num-buckets=1 --graph-path=/data/lisa/data/genomics/graph/pancan-tissue-graph.hdf5 --samples-path=/data/lisa/data/genomics/TCGA/TCGA_tissue_ppi.hdf5 --exp-name=genemania --cuda=True genemania

# Pancan => Genemania
# Kegg => Regnet
# Get the inputs
for i in "$@"
do
case $i in
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
    exp_name="${i#*=}"
    ;;
    --cuda=*)
    cuda="${i#*=}"
    ;;
esac
done

for bucket_idx in $(seq 1 $num_buckets)
do
    #sbatch --output $exp_dir"/slurm-%j.out" -x mila00 ./experiments/launch_scripts/launch_one_fig_4_job.sh $bucket_idx $num_buckets $exp_name $graph
    ./experiments/launch_scripts/launch_one_fig_4_job.sh --bucket-idx=$bucket_idx --num-buckets=$num_buckets --exp-name=$exp_name --graph-path=$graph_path --samples-path=$samples_path --cuda=$cuda
done
