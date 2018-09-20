#!/usr/bin/env bash

# Sometimes useful on the MILA cluster
# export HOME=`getent passwd $USER | cut -d':' -f6`
# source activate pytorch0.2

# EXAMPLES OF HOW TO CALL THIS FILE, note that you should run this from the root of the project
# genes=("RPL13" "HLA-B" "S100A9" "IFIT1" "RPL5" "RPS31" "ZFP82" "IL5" "DLGAP2")
# ./experiments/launch_scripts/launch_all_fig_5_jobs.sh --genes="${genes[*]}" --graph-path=/data/lisa/data/genomics/graph/pancan-tissue-graph.hdf5 --samples-path=/data/lisa/data/genomics/TCGA/TCGA_tissue_ppi.hdf5 --cuda=True

# Pancan => Genemania
# Kegg => Regnet
# Get the inputs

for i in "$@"
do
case $i in
    --genes=*)
    genes="${i#*=}"
    ;;
    --graph-path=*)
    graph_path="${i#*=}"
    ;;
    --samples-path=*)
    samples_path="${i#*=}"
    ;;
    --trials=*)
    trials="${i#*=}"
    ;;
    --cuda=*)
    cuda="${i#=}"
    ;;
esac
done

for gene in ${genes[@]}
do
    sbatch --output "experiments/results/slurm-%$gene.out" --gres=gpu -x mila00 ./experiments/launch_scripts/launch_one_fig_5_job.sh --gene=$gene --graph-path=$graph_path --samples-path=$samples_path --trials=$trials --cuda=$cuda
    #./experiments/launch_scripts/launch_one_fig_5_job.sh --gene=$gene --graph-path=$graph_path --samples-path=$samples_path --trials=$trials --cuda=$cuda
done
