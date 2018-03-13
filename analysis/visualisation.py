import os
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_weights(path, file_name='weights.pkl'):

    path = os.path.join(path, file_name)
    weights = pkl.load(open(path))
    return weights

def load_graph(path, file_name='graphs.pkl'):
    graphs = pkl.load(open(os.path.join(path, file_name)))
    return graphs


def load_representation(path, file_name='representations.pkl'):
    reps = pkl.load(open(os.path.join(path, file_name)))
    return reps

def load_gradients(path, file_name='grads.pkl'):
    grads = pkl.load(open(os.path.join(path, file_name)))
    return grads


def draw_graph(adj, node_color=None, pos=None):
    nb_nodes = adj.shape[0]

    order = np.arange(adj.shape[0]).reshape(int(np.sqrt(nb_nodes)), int(np.sqrt(nb_nodes)))

    # If we don't give the position, we assume it's a grid.
    if pos is None:
        pos = [np.argwhere(order == i)[0] for i in range(order.shape[0] * order.shape[1])]

    np.fill_diagonal(adj, 0.)
    G = nx.Graph(adj)

    if node_color is None:
        node_color = 'r'

    nx.draw_networkx(G, pos, nodelist=range(nb_nodes), node_color=node_color)
    #nx.draw(G, pos, nodelist=range(nb_nodes), node_color=node_color)
    # nx.draw_networkx(G)

    # nx.draw_networkx(G, pos, nodelist=list(order.flatten()))
    plt.show()

def get_interpretable_emb(model_state):

    emb = model_state['emb.emb']
    return emb

def vec_2d(emb, method='PCA'):
    emb = emb
    pca = PCA(n_components=2)
    return pca.fit_transform(emb)


def scatter_easily(emb, color=None, annotation=None):
    import matplotlib.pyplot as plt

    assert emb.shape[1] == 2

    if color is None:
        color = 'b'

    # Plotting the embeddings
    plt.scatter(emb[:, 0], emb[:, 1], c=color)

    if annotation is not None:
        for no, txt in annotation:
            plt.annotate(txt, (emb[no, 0], emb[no, 1]))

def visualize_all(representations, graph, no_ex, layer_to_show, grads=None):

    print "target: {}, prediction: {}".format(representations['example']['output'][no_ex], np.argmax(representations['logistic']['output'][no_ex]))
    print "Input:"
    draw_graph(graph, node_color=representations['example']['input'][no_ex][:, 0])

    for layer_name in layer_to_show:
        print "For {}".format(layer_name)

        node_color = (representations[layer_name]['output'][no_ex] ** 2).sum(axis=-1)
        if grads is not None:
            node_color = (grads[layer_name][no_ex] ** 2).sum(axis=-1)

        draw_graph(graph, node_color=node_color)

def extract_chain_representation(representations, graph, no_ex, layer_to_show, top_k=5, fn=None):

    if fn is None:
        fn = lambda layer_name, no_ex: (representations[layer_name]['output'][no_ex] ** 2).sum(axis=-1)

    top_node = []
    node_to_check = range(graph.shape[0])

    for layer_name in layer_to_show:
        layer = fn(layer_name, no_ex)[node_to_check]
        top = np.argsort(layer)[-top_k:]
        top = [node_to_check[node] for node in top]
        print "Top:", top
        top_node.append(top)

        neighbours = [list(np.where(graph[node])[0]) for node in top] + [top]
        neighbours = list(set([y for x in neighbours for y in x]))
        print "Next to consider: ", neighbours
        node_to_check = neighbours

    return top_node

def extract_chain_representation_2(representations, graph, no_ex, layer_to_show, top_k=5, fn=None):

    if fn is None:
        fn = lambda layer_name, no_ex: (representations[layer_name]['output'][no_ex] ** 2).sum(axis=-1)

    def extract_top_neighbours(node, values):
        graph[node][node] = 1.
        my_neighbours = graph[node] * values
        top_neighbours = np.argsort(my_neighbours)[-top_k:]
        top_neighbours = [n for n in top_neighbours if my_neighbours[n] > 0.]
        return list(top_neighbours)

    top_node = []

    # Get ours roots.
    layer = fn(layer_to_show[0], no_ex)
    roots = np.argsort(layer)[-1:]
    top_node.append(roots)

    # The rest of the layers
    for layer_name in layer_to_show[1:]:

        layer = fn(layer_name, no_ex)
        new_roots = []
        for r in roots:
            top = extract_top_neighbours(r, layer)
            new_roots = new_roots + top

        roots = list(set(new_roots))
        top_node.append(roots)
    return top_node