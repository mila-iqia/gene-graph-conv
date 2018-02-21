import os
import sys
from Bio.KEGG import REST
import cPickle as pkl
from tqdm import tqdm



def get_human_pathways(output_dir='raw_data'):
    '''
    Loads all human pathways and puts them into a dictionary 
    pathway_id -> pathway_file

    In case the data is not downloaded, downloads it from the KEGG database
    '''

    if not os.path.exists(os.path.join(output_dir, 'pathways.pkl')):
        human_pathways = REST.kegg_list("pathway", "hsa").read()
        pathways = []
        for line in human_pathways.rstrip().split("\n"):
            entry, description = line.split("\t")
            pathways.append(entry)
        
        with open(os.path.join(output_dir, 'pathways.pkl'), 'w') as fout:
            pkl.dump(pathways, fout)
    else:
        with open(os.path.join(output_dir, 'pathways.pkl')) as fin:
            pathways = pkl.load(fin)

    pathways_dict = {}
    for pathway in tqdm(pathways):
        if not os.path.exists(os.path.join(output_dir, pathway)):
            pathway_file = REST.kegg_get(pathway).read()
            with open(os.path.join(output_dir, pathway), 'w') as fout:
                pkl.dump(pathway_file, fout)
        else:
            with open(os.path.join(output_dir, pathway)) as fin:
                pathway_file = pkl.load(fin)
        pathways_dict[pathway] = pathway_file
    
    return pathways_dict


def get_human_regulation_TRRUST(output_dir='raw_data', remove_unknown=False):
    '''
    Gene regulation data extracted from text of papers using automated text parsing
    In case data is not preprocessed and downloaded, does it
    Outputs dict of gene ordered pairs and interactions types
    remove_unknown removes all regulations that are labeled as 'Unknown'
    '''

    data_file = os.path.join(output_dir, 'trrust_rawdata.human.tsv')
    processed_data_file = os.path.join(output_dir, 'trrust.human.pkl')
    if not os.path.exists(data_file):
        os.system('wget http://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv')
        os.system('mv trrust_rawdata.human.tsv '+data_file)
    else:
        if not os.path.exists(processed_data_file):
            regulation_data = {}
            regulation_types = set([])
            with open(data_file) as fin:
                for line in tqdm(fin):
                    gene1, gene2, reg_type, _ = line.split()
                    regulation_data[(gene1, gene2)] = reg_type
                    regulation_types.add(reg_type)

            with open(processed_data_file, 'w') as fout:
                pkl.dump((regulation_data, regulation_types), fout)
        else:
            with open(processed_data_file) as fin:
                regulation_data, regulation_types = pkl.load(fin)

    if remove_unknown:
        clean_regulation_data = {}
        for key in regulation_data.keys():
            if regulation_data[key] != 'Unknown':
                clean_regulation_data[key] = regulation_data[key]
        regulation_types.discard('Unknown')
        return clean_regulation_data, regulation_types

    return regulation_data, regulation_types

def get_human_regulation_RegNetwork(output_dir='raw_data'):
    # csv_file_node = os.path.join(output_dir, 'human.node')
    # csv_file_source = os.path.join(output_dir, 'human.source')
    # if (not os.path.exists(csv_file_node) ) or (not os.path.exists(csv_file_source)):
    #     os.system('wget http://www.regnetworkweb.org/download/human.zip')
    #     os.system('unzip human.zip -d '+output_dir)
    #     os.system('rm human.zip')

    # reg_file = os.path.join(output_dir, 'kegg.human.reg.direction')
    # if not os.path.exists(reg_file):
    #     os.system('wget http://www.regnetworkweb.org/download/RegulatoryDirections.rar')
    #     os.system('unrar e RegulatoryDirections.rar ' + output_dir)
    #     os.system('rm RegulatoryDirections.rar')
    
    # owl_file = os.path.join(output_dir, 'human.owl')
    # if not os.path.exists(owl_file):
    #     os.system('wget http://www.regnetworkweb.org/download/human.owl.zip')
    #     os.system('unzip human.owl.zip -d '+output_dir)
    #     os.system('rm human.owl.zip')

    # with open(csv_file_node) as fin:
    #     for line in fin:
    #         print line
    #         # break

    # with open(csv_file_source) as fin:
    #     for line in fin:
    #         print line
    #         break

    # with open(reg_file) as fin:
    #     for line in fin:
    #         print line
    #         break
    data = []
    csv_file = os.path.join(output_dir, 'export_Wed_Feb_21_18_09_14_UTC_2018.csv')
    with open(csv_file) as fin:
        for line in fin:
            a = line.split(',')
            gene1 = a[0][1:-1]
            gene2 = a[2][1:-1]
            data.append((gene1, gene2))
    return data

    #
    