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

## *test.py* 
This script takes two gene names and sample number and compares final numbers from the *.hdf5* file to the numbers from 
initial files. The correct test looks like this:

```
Checking graph
Gene name =  EGR1 Gene index =  3957
Gene name =  ZFP36 Gene index =  5165
Graph weight from dataset file =  0.024
Mapping from genemania  ENSG00000120738  to gene name  EGR1
Mapping from genemania  ENSG00000128016  to gene name  ZFP36
Processing category: Co-expression publication: Perou-Botstein-1999
Processing category: Co-expression publication: Perou-Botstein-2000


Checking expression
Dataset sample index =  500 gene a exp =  14.2618 gene b exp =  14.3929
Dataset sample name  TCGA-E9-A1NG-11
Raw data sample =  TCGA-E9-A1NG-11 Raw data gene name =  ZFP36 Raw data expression =  14.2618
Raw data sample =  TCGA-E9-A1NG-11 Raw data gene name =  EGR1 Raw data expression =  14.3929


Checking labels
Dataset sample label = [ 1.  0.  0.  0.  0.  0.  0.  0.]
Dataset sample name = Infiltrating Ductal Carcinoma
Raw data sample name =  TCGA-E9-A1NG-11 raw data label name =  Infiltrating Ductal Carcinoma
```
