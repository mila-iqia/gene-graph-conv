import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import cPickle as pkl
from tqdm import tqdm
import sys
import numpy as np

import h5py

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Process gene expression data.')	
	parser.add_argument('-dataset', default='BRCA_coexpr.hdf5', help='Dataset filename')
	parser.add_argument('-gene_a', default='ZFP36' , help='First gene name')
	parser.add_argument('-gene_b', default='EGR1', help='Second gene name')
	parser.add_argument('-sample_id', default='500', help='Expression sample')
	
	
	args = parser.parse_args()
	

	print '\n\nChecking graph'
	fmy = h5py.File(args.dataset,"r")
	gene_names = fmy["gene_names"]
	for gene_idx in xrange(gene_names.shape[0]):
		if gene_names[gene_idx] == args.gene_a:
			gene_a_idx = gene_idx
		if gene_names[gene_idx] == args.gene_b:
			gene_b_idx = gene_idx
		if gene_names[gene_idx] == args.gene_b or gene_names[gene_idx] == args.gene_a:
			print 'Gene name = ', gene_names[gene_idx], 'Gene index = ', gene_idx 

	weight = fmy["graph_data"][gene_a_idx][gene_b_idx]
	print 'Graph weight from dataset file = ', weight

	with open("mappings/identifier_mappings.txt") as fin:
		for n, line in enumerate(fin):
			sline = line.split()
			graph_id = sline[0]
			gene_name = sline[1]
			if gene_name == args.gene_a:
				gene_a_gid = graph_id
				print 'Mapping from genemania ', graph_id, ' to gene name ', gene_name
			
			if gene_name == args.gene_b:
				gene_b_gid = graph_id
				print 'Mapping from genemania ', graph_id, ' to gene name ', gene_name
	

	pub_cat = ['Co-expression']
	pub_subset = ['Perou-Botstein-1999', 'Perou-Botstein-2000']
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
				weight = line.split()[2]
				if geneA == gene_a_gid and geneB == gene_b_gid:
					print 'Gene A id =', geneA, 'gene B id =', geneB, 'Weigh =', weight

	print '\n\nChecking expression'
	sample_id = int(args.sample_id)
	exp_gene_a = fmy["expression_data"][sample_id, gene_a_idx]
	exp_gene_b = fmy["expression_data"][sample_id, gene_b_idx]
	print 'Dataset sample index = ', sample_id, 'gene a exp = ', exp_gene_a, 'gene b exp = ', exp_gene_b
	sample_name = fmy["sample_names"][sample_id]
	print 'Dataset sample name ', sample_name
	
	with open("expressionData/BRCA_HiSeqV2") as fin:
		samples = fin.readline().split()[1:]
		sample_col_num = samples.index(sample_name)
		for n, line in enumerate(fin):
			sline = line.split()
			gene_id = sline[0]
			expr_data = sline[1:]
			if gene_id == args.gene_a or gene_id == args.gene_b:
				print 'Raw data sample = ', sample_name, 'Raw data gene name = ', gene_id, 'Raw data expression = ', expr_data[sample_col_num]
	

	print '\n\nChecking labels'
	print 'Dataset sample label =', fmy["labels_data"][sample_id]
	label_idx = np.argmax(fmy["labels_data"][sample_id])
	print 'Dataset sample name =', fmy["labels_data"].attrs[str(label_idx)]


	label_col_name = 'histological_type'
	sample_col_name = 'sampleID'
	with open("expressionData/BRCA_clinicalMatrix") as fin:
		header = fin.readline().split('\t')
		label_id_idx = header.index(label_col_name)
		sample_id_idx = header.index(sample_col_name)
		for n, line in enumerate(fin):
			sline = line.split('\t')
			data_label_id = sline[label_id_idx]
			data_sample_id = sline[sample_id_idx]
			
			if data_sample_id == sample_name:
				print 'Raw data sample name = ', data_sample_id, 'raw data label name = ', data_label_id

	fmy.close()
	