import torch
import numpy as np
import models
import pickle
import os

def feature_selection(model, dataset, opt, top=100):

    """
    Save the norm of the weight of the last layer.
    :param model:
    :param dataset:
    :param opt:
    :param top:
    :return:
    """

    top_features = [] # TODO: export emb.
    if opt.attention_layer > 0:
        print "Feature selection when using attention is not implemented yet."
        return top_features

    try:
        for i, layer in enumerate(model.my_logistic_layers):
            weight = layer.weight.data

            top_features.append({})

            for no_class in range(weight.size(0)):
                nb_channel = 1 if weight[no_class].size(0) == model.nb_nodes else model.nb_channels[0]
                #import ipdb; ipdb.set_trace()
                this_layer_feature = torch.abs(weight[no_class].view(nb_channel, model.nb_nodes)).sum(0) # It's a logistic regression, so lets do that.

                _, top_k = torch.topk(this_layer_feature, top)
                top_k_names = dataset.node_names[top_k.cpu().numpy()]

                top_features[i][dataset.labels_name(no_class)] = (top_k_names, top_k.cpu().numpy())
    except AttributeError:
        print "{} doesn't have any logistic layers.".format(model)

    return top_features

def get_graph(model):

    """
    Get the graph of a model.
    :param model:
    :return:
    """

    retn = []

    if not issubclass(model.__class__, models.GraphNetwork):
        print "The model is not a graph convolution."
        return retn

    # Go over all convolution
    for conv_layer in model.my_convs:
        adj = conv_layer.adj
        to_keep = conv_layer.to_keep
        retn.append([adj, to_keep])

    return retn

def monitor_everything(model, dataset, opt, exp_dir):
    print "Extracting the important features..."
    features = feature_selection(model, dataset, opt)
    pickle.dump(features, open(os.path.join(exp_dir, 'features.pkl'), 'wb'))

    print "Extracring the graphs..."
    graphs = get_graph(model)
    pickle.dump(graphs, open(os.path.join(exp_dir, 'graphs.pkl'), 'wb'))

def load_everything(exp_dir):

    print "Loading the data..."
    features = pickle.load(open(os.path.join(exp_dir, 'features.pkl')))
    graphs = pickle.load(open(os.path.join(exp_dir, 'graphs.pkl')))
    print "Done!"

    return features, graphs
