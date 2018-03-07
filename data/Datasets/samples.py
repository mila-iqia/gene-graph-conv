import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import cPickle as pkl
from tqdm import tqdm
import sys
import numpy as np

import h5py

def map_samples(cliMat_filename = "expressionData/BRCA_clinicalMatrix",
				label_col_name = 'histological_type',
				sample_col_name = 'sampleID',
				output_map_filename = 'mappings/sample2index.pkl'
				):
	#set of samples and mapping to index
	set_samples = set([])
	with open(cliMat_filename) as fin:
		header = fin.readline().split('\t')
		label_id_idx = header.index(label_col_name)
		sample_id_idx = header.index(sample_col_name)
		for n, line in enumerate(fin):
			sline = line.split('\t')
			label_id = sline[label_id_idx]
			sample_id = sline[sample_id_idx]
			if sample_id.find('TCGA')!=-1:
				set_samples.add(sample_id)
	
	#mapping sample name to index
	sample2index = {}
	for idx,sample in enumerate(set_samples):
		sample2index[sample] = idx

	print 'Number of samples:', len(sample2index.keys())

	with open(output_map_filename, 'w') as fout:
		pkl.dump(sample2index, fout)

	return sample2index


def extract_labels(	sample2index_filename = 'mappings/sample2index.pkl',
					cliMat_filepath = "expressionData/BRCA_clinicalMatrix",
					label_col_name = 'histological_type',
					sample_col_name = 'sampleID'):
	with open(sample2index_filename, 'r') as fin:
		sample2index = pkl.load(fin)

	#getting labels set
	labels_set = set([])
	with open(cliMat_filepath) as fin:
		header = fin.readline().split('\t')
		label_id_idx = header.index(label_col_name)
		sample_id_idx = header.index(sample_col_name)
		for n, line in enumerate(fin):
			sline = line.split('\t')
			label_id = sline[label_id_idx]
			sample_id = sline[sample_id_idx]
			if label_id=='' or label_id=='Other, specify':
				label_id = 'Other'
			if sample_id in sample2index.keys():
				labels_set.add(label_id)

	labels_set = list(labels_set)
	print 'Labels:', labels_set
	
	D = len(labels_set)
	N = len(sample2index.keys())
	print 'Compiling the labels dataset.'
	print 'Num labels:', D
	print 'Number of samples:', N

	labels_mat = np.zeros((N, D))

	with open(cliMat_filepath) as fin:
		header = fin.readline().split('\t')
		label_id_idx = header.index(label_col_name)
		sample_id_idx = header.index(sample_col_name)
		for n, line in enumerate(fin):
			sline = line.split('\t')
			label_id = sline[label_id_idx]
			sample_id = sline[sample_id_idx]
			if label_id=='' or label_id=='Other, specify':
				label_id = 'Other'
			if sample_id in sample2index.keys():
				
				label_index = labels_set.index(label_id)
				labels_mat[sample2index[sample_id], label_index] = 1.0
	
	return labels_mat, labels_set


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Process gene expression data.')
	parser.add_argument('--dataset', help='Dataset filename')
	parser.add_argument('--map_sample2index', help='Map samples to index')
	parser.add_argument('--extract_labels', help='Get dataset labels')
	args = parser.parse_args()


	if not args.map_sample2index is None:
		sample2index = map_samples()

		if not args.dataset is None:
			#Writing mapping to the dataset
			N = len(sample2index.keys())
			fmy = h5py.File(args.dataset,"a")
			
			try:
				sample_names = fmy.create_dataset("sample_names", (N,), dtype="S64")
			except:
				del fmy["sample_names"]
				sample_names = fmy.create_dataset("sample_names", (N,), dtype="S64")	

			for sample_name in sample2index.keys():
				sample_names[sample2index[sample_name]] = sample_name
			fmy.flush()
			fmy.close()

	
	if not args.extract_labels is None:
		labels_mat, labels_set = extract_labels()
		N, D = labels_mat.shape
		
		if not args.dataset is None:
			fmy = h5py.File(args.dataset,"a")
			try:
				labels_data = fmy.create_dataset("labels_data", (N,D), dtype=np.dtype('float32'))
			except:
				del fmy["labels_data"]
				labels_data = fmy.create_dataset("labels_data", (N,D), dtype=np.dtype('float32'))
			
			#Labels description
			for i, label_name in enumerate(labels_set):
				labels_data.attrs[label_name] = i
				labels_data.attrs[str(i)] = label_name

			#Moving data to hdf5
			for i in tqdm(xrange(labels_mat.shape[0])):
				labels_data[i] = labels_mat[i,:]

			fmy.flush()
			fmy.close()