# Dataset construction
This directory contains scripts to construct datasets with gene expression and corresponding graph.
The major script is *run.sh*. It takes care of all the procedures to make BRCA_coexpr dataset. Before running 
this script create the following directories:
- "mappings"
- "expressionData"
- "data"
and get gene synonyms annotations *hugo_gencode_v24_gtf* and place it in *mappings* directory.

# Description

## *download.py* 
This script downloads all the neccessary files and places them in previously created directories. Currently
I do not download *hugo_gencode_v24_gtf*, so you need to find it manually.

## *graph.py* 
This file contains procedures to map from genemania node ids to normal gene names and to index all the genes in the graph.

## *samples.py* 
This file contains procedures to map from sample name to integer index and to make labels.

## *main_data.py* 
Here are the functions that rearrange graph and expression main data using previously constructed mappings from 
gene nodes to indexes and from samples to indexes. Also, used to make an empty *.hdf5* file.
