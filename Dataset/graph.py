import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import cPickle as pkl
from tqdm import tqdm
import sys
import numpy as np

import h5py

def map_graph2data( data_genes_filename = "mappings/hugo_gencode_v24_gtf",
					graph_genes_filename = "mappings/identifier_mappings.txt",
					output_filename = 'mappings/preprocessed_mappings.pkl'):
	data_gene_names = set([])
	with open(data_genes_filename) as fin:
		for n, line in enumerate(fin):
			ident = line.split()[0]
			gene_name = line.split()[1]
			data_gene_names.add(gene_name)
	print 'Number of gene names in data:', len(data_gene_names)

	graph_gene_id = {}
	with open(graph_genes_filename) as fin:
		for n, line in enumerate(fin):
			sline = line.split()
			graph_id = sline[0]
			gene_name = sline[1]
			if sline[2]=='Gene' or sline[2]=='Synonym':
				graph_gene_id[gene_name] = graph_id
	print 'Number of gene ids in graph:', len(graph_gene_id)

	map_data2graph = {}
	map_graph2data = {}
	for data_id in tqdm(data_gene_names):
		if data_id in graph_gene_id.keys():
			graph_id = graph_gene_id[data_id]
			map_data2graph[data_id] = graph_id
			map_graph2data[graph_id] = data_id
		else:
			pass
	
	print 'D2G and G2D keys length:', len(map_data2graph), len(map_graph2data)
	with open(output_filename, 'w') as fout:
		pkl.dump((map_data2graph, map_graph2data), fout)


def map_gene2index(	graph2data_filename = 'mappings/preprocessed_mappings.pkl',
					pub_cat = None,
					pub_subset = None,
					output_filename = 'mappings/gene2index.pkl'):
	
	with open(graph2data_filename, 'r') as fin:
		map_data2graph, map_graph2data = pkl.load(fin)

	#getting set of gene identificators in the graph data
	set_graph_ids = set([])
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
				set_graph_ids.add(geneA)
				set_graph_ids.add(geneB)

	print 'Number of gene identifiers in graph', len(set_graph_ids)

	#converting graph_ids to gene names
	set_graph_genes = set([])
	for graph_id in tqdm(set_graph_ids):
		if graph_id in map_graph2data.keys():
			set_graph_genes.add(map_graph2data[graph_id])
	print 'Number of gene names in graph', len(set_graph_genes)

	#mapping gene name to index
	gene2index = {}
	for idx,gene in enumerate(set_graph_genes):
		gene2index[gene] = idx

	with open('mappings/gene2index.pkl', 'w') as fout:
		pkl.dump(gene2index, fout)
	
	return gene2index

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Process gene names graph to data mappings.')
	parser.add_argument('--dataset', help='Dataset filename')
	parser.add_argument('--map_graph2data', help='Process graph and data mapping')
	parser.add_argument('--map_gene2index', help='Map graph genes to indexes')
	args = parser.parse_args()

	if not args.map_graph2data is None:
		map_graph2data()		

	if not args.map_gene2index is None:
		
		gene2index = map_gene2index(pub_cat = ['Co-expression'],
									pub_subset = ['Perou-Botstein-1999', 'Perou-Botstein-2000'])
		
		if not args.dataset is None:
			#Writing mapping to the dataset
			M = len(gene2index.keys())
			fmy = h5py.File(args.dataset,"a")
			
			try:
				gene_names = fmy.create_dataset("gene_names", (M,), dtype="S64")
			except:
				del fmy["gene_names"]
				gene_names = fmy.create_dataset("gene_names", (M,), dtype="S64")

			for gene_name in gene2index.keys():
				gene_names[gene2index[gene_name]] = gene_name
			fmy.flush()
			fmy.close()

	


	