import os
import csv
import pickle
from gtfparse import read_gtf
import random


def record_result(results, experiment, filename):
    results = results.append(experiment, ignore_index=True)
    results_dir = "/".join(filename.split('/')[0:-1])

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    pickle.dump(results, open(filename, "wb"))
    return results

def symbol_map(gene_symbols):
    """
    This gene code map was generated on February 18th, 2019
    at this URL: https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=gd_prev_sym&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit
    it enables us to map the gene names to the newest version of the gene labels
    """
    filename = os.path.join(os.path.dirname(__file__), 'genenames_code_map_Feb2019.txt')
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        x = {row[0]: row[1] for row in csv_reader}

        map = {}
        for key, val in x.items():
            for v in val.split(", "):
                if key not in gene_symbols:
                    map[v] = key
    return map


def ncbi_to_hugo_map(gene_symbols, datastore="./data"):
    with open(datastore + '/graphs/enterez_NCBI_to_hugo_gene_symbol_march_2019.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        x = {int(row[1]): row[0] for row in csv_reader if row[1] != ""}

        map = {}
        for key, val in x.items():
            map[key] = val
    return map


def ensg_to_hugo_map(datastore="./data"):
    with open(datastore + "/datastore/ensembl_map.txt") as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensmap = {row[1]: row[0] for row in csv_reader if row[0] != ""}

    return ensmap


def ensp_to_hugo_map(datastore="./data"):
    """
    You should download the file Homo_sapiens.GRCh38.95.gtf from :
    ftp://ftp.ensembl.org/pub/release-95/gtf/homo_sapiens/Homo_sapiens.GRCh38.95.gtf.gz

    Store the file in datastore
    """
    savefile = datastore + "/datastore/ensp_ensg_df.pkl"

    # If df is already stored, return the corresponding dictionary
    if os.path.isfile(savefile):
        f = open(savefile, 'rb')
        df = pickle.load(f)
        f.close()
    else:
        df = read_gtf(datastore + "/datastore/Homo_sapiens.GRCh38.95.gtf")
        df = df[df['protein_id'] != ''][['gene_id', 'protein_id']].drop_duplicates()
        df.to_pickle(savefile)

    # ENSG to hugo map
    with open(datastore + "/datastore/ensembl_map.txt") as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensg_map = {row[1]: row[0] for row in csv_reader if row[0] != ""}

    # ENSP to hugo map
    ensmap = {}
    for index, row in df.iterrows():
        if row['gene_id'] in ensg_map.keys():
            ensmap[row['protein_id']] = ensg_map[row['gene_id']]

    return ensmap


def randmap(nodelist, seed=0):
    """
    :param nodelist: nx_graph.nodes
    :return: permutation map on the set of nodes of the graph
    """
    nodes = list(nodelist)
    randnodes = nodes.copy()
    random.seed(seed)
    random.shuffle(randnodes)
    return dict(zip(nodes, randnodes))

def get_file_separator(filename):
    separators = {'.tsv' : '\t', '.txt': '\t', '.csv': ','}
    sep = separators[os.path.splitext(filename.replace('.gz', ''))[-1]]
    return sep

