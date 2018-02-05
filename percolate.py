import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque
from tqdm import tqdm
import argparse
import h5py

def f(p):
    return np.random.binomial(1, p)

def sq2d_lattice_graph(x_size, y_size, value_fn):
    G = nx.Graph()
    #adding nodes
    nodeinorder=[]
    for i in xrange(x_size):
        for j in xrange(y_size):
            val = value_fn()
            G.add_node((i,j), value=val)
            nodeinorder.append((i,j))

    #adding edges
    for node in G.nodes():
        if node[0]>0:
            G.add_edge((node[0]-1, node[1]), node)
        if node[0]<(x_size-1):
            G.add_edge((node[0]+1, node[1]), node)
        if node[1]>0:
            G.add_edge((node[0], node[1]-1), node)
        if node[1]<(y_size-1):
            G.add_edge((node[0], node[1]+1), node)
    return G, nodeinorder

def get_density(G):
    vals = dict(nx.get_node_attributes(G, 'value'))
    num_on = np.sum( list(vals.values()) )
    num_total = len(list(vals.values()))
    density = float(num_on)/float(num_total)
    return density

def if_percolates_simple(G, x_size, y_size):
    T = G.copy()
    for data_node in G.nodes.data():
        if data_node[1]['value'] == 0:
            T.remove_node(data_node[0])
        
    source = []
    ground = []
    M = T.copy()
    
    for node in T.nodes():
        if node[0] == 0:
            new_node = (-1, node[1])
            M.add_node(new_node, value = 2)
            M.add_edge(new_node, node)
            ground.append(new_node)
        if node[0] == (x_size-1):
            new_node = (x_size, node[1])
            M.add_node(new_node, value = 2)
            M.add_edge(new_node, node)
            source.append(new_node)
    
    
    # detect path through 1-s
    for s_node in ground:
        for u, v in nx.dfs_edges(M, s_node):
            if v in source:
                return True
   
    return False

def sq2d_lattice_percolation_simple(size_x=10, size_y=10, prob=0.3):
    def fp(): return f(prob)

    #Generating square lattice graph
    G, nio = sq2d_lattice_graph(size_x,size_y, fp)
    G_0 = G.copy()
    
    #Getting density of open nodes
    density = get_density(G)

    #Checking percolation
    perc = if_percolates_simple(G, size_x, size_y)
    
    upper_density_threshold = 0.51
    lower_density_threshold = 0.49

    #correcting density
    while( density<lower_density_threshold or density>upper_density_threshold):
        G_new = G.copy()
        node = random.choice(list(G_new.nodes))
        
        if density<lower_density_threshold:
            if G_new.nodes[node]['value'] == 1:
                continue
            G_new.nodes[node]['value'] = 1

        if density>upper_density_threshold:
            if G_new.nodes[node]['value'] == 0:
                continue
            G_new.nodes[node]['value'] = 0


        perc_new = if_percolates_simple(G_new, size_x, size_y)
        if perc == perc_new:
            G = G_new.copy()
            density = get_density(G)

    return G, G_0, perc, density, nio


def sq2d_plot_graph(G):
    positionsG = {}
    for node in G.nodes():
        positionsG[node] = np.array([node[0],node[1]], dtype='float32')

    labelsG = dict(nx.get_node_attributes(G, 'value'))
    optionsG = {
        'node_size': 100,
        'width': 3,
        'with_labels':False
    }
    nx.draw_networkx(G, pos = positionsG, nodelist=list(labelsG.keys()), node_color=list(labelsG.values()), **optionsG)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Percolation dataset')
    parser.add_argument('--dataset', help='Dataset filename')
    parser.add_argument('--test', type=int, help='Generate example')
    parser.add_argument('--N', type=int, default = 100, help='Number of examples')
    parser.add_argument('--size_x', type=int, default = 16, help='X dim size')
    parser.add_argument('--size_y', type=int, default = 16, help='Y dim size')
    parser.add_argument('--prob', type=float, default = 0.562, help='On/off probability')


    args = parser.parse_args()

    if not args.test is None:
        if args.dataset is None:
            try:
                G, T, perc, dens, nio = sq2d_lattice_percolation_simple(size_x=10, size_y=10, prob = 0.562)
            except:
                print 'Try again'
                sys.exit(0)
            print 'Percolation = ', perc, 'Density = ', dens

            plt.subplot(121)
            sq2d_plot_graph(G)

            plt.subplot(122)
            sq2d_plot_graph(T)
            plt.show()
        else:
            fmy = h5py.File(args.dataset,"r")

            mat = np.array(fmy["graph_data"])
            G = nx.from_numpy_matrix(mat)
            nodes_attr = {}
            for i,node in enumerate(G.nodes()):
                nodes_attr[node] = fmy["expression_data"][args.test][i]
            nx.set_node_attributes(G, nodes_attr, 'value')

            plt.figure()
            labelsG = dict(nx.get_node_attributes(G, 'value'))

            nodeinorder=[]
            for x in xrange(args.size_x):
                for y in xrange(args.size_y):
                    nodeinorder.append((y,x))
            
            nx.draw_networkx(G, pos=nodeinorder,with_labels=True,nodelist=list(labelsG.keys()), node_color=list(labelsG.values()))
            plt.show()

            print 'Label = ', fmy["labels_data"][args.test]

            fmy.close()


    if (not args.dataset is None) and (args.test is None):
        if os.path.exists(args.dataset):
            os.remove(args.dataset)
        #Moving data to hdf5
        fmy = h5py.File(args.dataset,"w")

        #generate graph
        G, T, perc, dens, nio = sq2d_lattice_percolation_simple( args.size_x, args.size_y, args.prob)
        node_list = nio#list(G.nodes())
        
        M = len(node_list)
        mat = nx.adjacency_matrix(G, nodelist=node_list).todense()
        # mat = nx.to_numpy_matrix(nx.adjacency_matrix(G), weight=None)

        graph_data = fmy.create_dataset("graph_data", (M,M), dtype=np.dtype('float32'))
        for i in xrange(M):
            graph_data[i] = mat[i,:]

        expression_data = fmy.create_dataset("expression_data", (args.N,M), dtype=np.dtype('float32'))
        labels_data = fmy.create_dataset("labels_data", (args.N,), dtype=np.dtype('float32'))

        for i in tqdm(xrange(args.N)):

            if i%2 == 0: #generate positive example
                perc = False
                while perc == False:
                    G, T, perc, dens, nio = sq2d_lattice_percolation_simple( args.size_x, args.size_y, args.prob)
                attrs = nx.get_node_attributes(G, 'value')
                features = np.zeros((M,), dtype='float32')
                for j,node in enumerate(node_list):
                    features[j] = attrs[node]
                expression_data[i] = features
                labels_data[i] = 1

            else: #generate negative example
                perc = True
                while perc == True:
                    G, T, perc, dens, nio = sq2d_lattice_percolation_simple( args.size_x, args.size_y, args.prob)
                attrs = nx.get_node_attributes(G, 'value')
                features = np.zeros((M,), dtype='float32')
                for j,node in enumerate(node_list):
                    features[j] = attrs[node]
                expression_data[i] = features
                labels_data[i] = 0

        fmy.flush()
        fmy.close()

