import urllib2
import urllib
from HTMLParser import HTMLParser
import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import cPickle as pkl

import gzip
import shutil

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
	list_datafiles = []
	def handle_starttag(self, tag, attrs):
		if tag == 'a':
			print "Encountered a start tag:", tag, attrs
			if attrs[0][1].find('Physical_Interactions')!=-1:
				self.list_datafiles.append(attrs[0][1])
				
			elif attrs[0][1].find('Co-expression')!=-1:
				self.list_datafiles.append(attrs[0][1])


	def handle_endtag(self, tag):
		if tag == 'a':
			print "Encountered an end tag :", tag

	def handle_data(self, data):
		pass

def retrieve(url, filepath, ungzip=False):
	if not ungzip:
		if not os.path.exists(filepath):
			urllib.urlretrieve(url, filepath)
	else:
		if not os.path.exists(filepath):
			filename = url[url.rfind('/')+1:]
			os.system('wget %s'%url)
			with gzip.open(filename, 'rb') as f_in, open(filepath, 'w') as f_out:
				shutil.copyfileobj(f_in, f_out)
			os.system('rm %s'%filename)

	


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Download and process gene interaction network.')
	parser.add_argument('--download_graph', help='Download graph data')
	parser.add_argument('--download_matrix', help='Download clinical matrix')
	parser.add_argument('--download_expression', help='Download expression data')
	parser.add_argument('--download_gene_loc', help='Download mapping of gene to chromosome')
	args = parser.parse_args()
	
	if not args.download_graph is None:
		URL = 'http://genemania.org/data/current/Homo_sapiens/'
		parser = MyHTMLParser()
		response = urllib2.urlopen(URL)
		html = response.read()
		parser.feed(html)

		for filename in parser.list_datafiles:
			print 'Downloading ', filename
			file_url = URL+filename
			filepath = os.path.join('data', filename)
			if not os.path.exists(filepath):
				urllib.urlretrieve(file_url, filepath)

		print 'Downloading identifier mappings'
		mappings_url = 'http://genemania.org/data/current/Homo_sapiens/identifier_mappings.txt'
		filepath = os.path.join('mappings', 'identifier_mappings.txt')
		if not os.path.exists(filepath):
			urllib.urlretrieve(mappings_url, filepath)

	if not args.download_matrix is None:
		URL = 'https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix.gz'
		retrieve(URL, 'expressionData/BRCA_clinicalMatrix', ungzip = True)

	if not args.download_expression is None:
		URL = 'https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz'
		retrieve(URL, 'expressionData/BRCA_HiSeqV2', ungzip = True)

	if not args.download_gene_loc is None:
		raise NotImplementedError
		URL = "ftp://ftp.sanger.ac.uk/pub/gencode/Gencode_human/release_24/gencode.v24.annotation.gtf.gz"
		retrieve(URL, 'mappings/hugo_gencode_v24_gtf_1', ungzip = True)
