#!/usr/bin/env bash

# Get the inputs
export HOME=`getent passwd $USER | cut -d':' -f6`
source activate pytorch0.2

# ./gene_inference/launch_all_my_stuff.sh  10 gene_inference/genemania /data/lisa/data/genomics/graph/pancan-tissue-graph.hdf5.hdf5
# ./gene_inference/launch_all_my_stuff.sh  10 gene_inference/pathway /data/lisa/data/genomics/graph/pathway_commons.hdf5
# ./gene_inference/launch_all_my_stuff.sh  10 gene_inference/pathway /data/lisa/data/genomics/graph/kegg.hdf5


num_buckets=$1
exp_dir=$2
graph_path=$3

for bucket in $(seq 1 $num_buckets)
do
    sbatch --output $exp_dir"/slurm-%j.out" -x mila00 ./gene_inference/launch_one_specific_job.sh $bucket $exp_dir $graph_path
    #./launch_one_specific_job.sh $bucket $exp_dir
done
