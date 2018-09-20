#!/usr/bin/env bash

# Sometimes useful on the MILA cluster
# export HOME=`getent passwd $USER | cut -d':' -f6`
# source activate pytorch0.2

# EXAMPLES OF HOW TO CALL THIS FILE, note that you should run this from the root of the project
# ./experiments/launch_scripts/launch_all_fig_4_jobs.sh 1 fig_4_genemania genemania
# ./experiments/launch_scripts/launch_all_fig_4_jobs.sh 1 fig_4_regnet regnet
# on Mila Cluster ./experiments/launch_scripts/launch_all_fig_4_jobs.sh 1 fig_4_genemania --graph-path=/data/lisa/data/genomics/graph/pancan-tissue-graph.hdf5 --samples-path=/data/lisa/data/genomics/TCGA/TCGA_tissue_ppi.hdf5 --cuda=True

# Pancan => Genemania
# Kegg => Regnet
# Get the inputs
for i in "$@"
do
case $i in
    --buckets=*)
    buckets="${i#*=}"
    ;;
    --graph-path=*)
    graph_path="${i#*=}"
    ;;
    --samples-path=*)
    samples_path="${i#*=}"
    ;;
    --exp-name=*)
    exp_name="${i#=}"
    ;;
    --cuda=*)
    cuda="${i#=}"
    ;;
esac
done

for bucket_idx in $(seq 1 $buckets)
do
    #sbatch --output $exp_dir"/slurm-%j.out" -x mila00 ./experiments/launch_scripts/launch_one_fig_4_job.sh $bucket_idx $num_buckets $exp_name $graph
    ./experiments/launch_scripts/launch_one_fig_4_job.sh $bucket_idx $num_buckets $exp_name $graph_path $samples_path $cuda
done
