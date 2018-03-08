import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import cPickle as pkl
from tqdm import tqdm
import sys
import numpy as np

import h5py

def extract_expression_data(graph2data_filename = 'mappings/preprocessed_mappings.pkl',
							gene2index_filename = 'mappings/gene2index.pkl',
							sample2index_filename = 'mappings/sample2index.pkl',
							expression_filename = "expressionData/BRCA_HiSeqV2"):
	with open(graph2data_filename, 'r') as fin:
		map_data2graph, map_graph2data = pkl.load(fin)
	with open(gene2index_filename, 'r') as fin:
		gene2index = pkl.load(fin)
	with open(sample2index_filename, 'r') as fin:
		sample2index = pkl.load(fin)

	N = len(sample2index.keys())
	M = len(gene2index.keys())
	print 'Compiling the expression dataset.'
	print 'Number of genes:', M
	print 'Number of samples:', N

	
	#reading data to matrix
	data_mat = np.zeros((N,M))
	with open(expression_filename) as fin:	
		samples = fin.readline().split()[1:]
		for n, line in enumerate(tqdm(fin)):
			sline = line.split()
			gene_id = sline[0]
			if gene_id in gene2index.keys():
				for j, value in enumerate(sline[1:]):
					sample_id = samples[j]
					if sample_id in sample2index.keys():
						data_mat[ sample2index[sample_id], gene2index[gene_id] ] = float(value)


	return data_mat

def extract_graph_data(		graph2data_filename = 'mappings/preprocessed_mappings.pkl',
							gene2index_filename = 'mappings/gene2index.pkl',
							sample2index_filename = 'mappings/sample2index.pkl',
							pub_cat = None,
							pub_subset = None):

	with open(graph2data_filename, 'r') as fin:
		map_data2graph, map_graph2data = pkl.load(fin)
	with open(gene2index_filename, 'r') as fin:
		gene2index = pkl.load(fin)
	with open(sample2index_filename, 'r') as fin:
		sample2index = pkl.load(fin)

	N = len(sample2index.keys())
	M = len(gene2index.keys())
	print 'Compiling the graph dataset.'
	print 'Number of genes:', M
	print 'Number of samples:', N

	
	#Loading data to the matrix
	graph_mat = np.zeros((M,M))
	data_dir = 'data'
	for filename in os.listdir(data_dir):
		category = filename[:filename.find('.')]
		publication = filename[filename.find('.')+1:filename.rfind('.')]
		
		if not pub_cat is None:
			if not category in pub_cat:
				continue
		if not pub_subset is None:
			if not publication in pub_subset:
				continue

		print 'Processing category:', category, 'publication:', publication
		file_path = os.path.join(data_dir, filename)
		with open(file_path, 'r') as fin:
			header = fin.readline()
			for line in fin:
				geneA = line.split()[0]
				geneB = line.split()[1]
				if geneA in map_graph2data.keys() and geneB in map_graph2data.keys():
					if map_graph2data[geneA] in gene2index.keys() and map_graph2data[geneB] in gene2index.keys():
						geneA_index = gene2index[map_graph2data[geneA]]
						geneB_index = gene2index[map_graph2data[geneB]]
						weight = float(line.split()[2])
						graph_mat[geneA_index, geneB_index] += weight
						graph_mat[geneB_index, geneA_index] += weight

	
	return graph_mat


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Process gene expression data.')
	parser.add_argument('--dataset', help='Dataset filename')
	parser.add_argument('--create_dataset', help='Create new dataset file')
	parser.add_argument('--expression_dataset', help='Compile the expression dataset')
	parser.add_argument('--graph_dataset', help='Add graph to dataset')
	
	args = parser.parse_args()

	if not args.create_dataset is None:
		if not args.dataset is None:
			fmy = h5py.File(args.dataset, "w")
			fmy.flush()
			fmy.close()
		else:
			print 'Enter the dataset name'
			sys.exit()

	if not args.expression_dataset is None:
		
		data_mat = extract_expression_data()
		N,M = data_mat.shape

		if not args.dataset is None:

			#Moving data to hdf5	
			fmy = h5py.File(args.dataset,"a")
			
			try:
				expression_data = fmy.create_dataset("expression_data", (N,M), dtype=np.dtype('float32'))
			except:
				del fmy["expression_data"]
				expression_data = fmy.create_dataset("expression_data", (N,M), dtype=np.dtype('float32'))
						
			for i in tqdm(xrange(N)):
				expression_data[i] = data_mat[i,:]
			
			fmy.flush()
			fmy.close()

	if not args.graph_dataset is None:
		
		graph_mat = extract_graph_data(	pub_cat = ['Co-expression'],
										pub_subset = ['Perou-Botstein-1999', 'Perou-Botstein-2000'])
		M = graph_mat.shape[0]

		if not args.dataset is None:
			fmy = h5py.File(args.dataset,"a")
			
			try:
				graph_data = fmy.create_dataset("graph_data", (M,M), dtype=np.dtype('float32'))
			except:
				del fmy["graph_data"]
				graph_data = fmy.create_dataset("graph_data", (M,M), dtype=np.dtype('float32'))
			
			#Moving data to hdf5
			for i in tqdm(xrange(M)):
				graph_data[i] = graph_mat[i,:]

			fmy.flush()
			fmy.close()

	