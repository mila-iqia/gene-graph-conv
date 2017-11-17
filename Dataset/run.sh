python download.py \
--download_graph 1 \
--download_matrix 1 \
--download_expression 1

python main_data.py \
--dataset BRCA_coexpr.hdf5 \
--create_dataset 1

python samples.py \
--dataset BRCA_coexpr.hdf5 \
--map_sample2index 1 \
--extract_labels 1

python graph.py \
--dataset BRCA_coexpr.hdf5 \
--map_graph2data 1 \
--map_gene2index 1

python main_data.py \
--dataset BRCA_coexpr.hdf5 \
--expression_dataset 1 \
--graph_dataset 1
