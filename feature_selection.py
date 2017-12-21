import torch
import numpy as np

def feature_selection(model, dataset, opt, top=25):

    top_features = []
    if opt.attention_layer > 0:
        print "Feature selection when using attention is implemented yet."
        return top_features

    try:
        for i, layer in enumerate(model.my_logistic_layers):
            weight = layer.weight.data

            top_features.append({})

            for no_class in range(weight.size(0)):
                nb_channel = 1 if weight[no_class].size(0) == model.nb_nodes else model.nb_channels[0]
                this_layer_feature = torch.abs(weight[no_class].view(nb_channel, model.nb_nodes)).sum(0) # It's a logistic regression, so lets do that.

                _, top_k = torch.topk(this_layer_feature, top)
                top_k_names = dataset.node_names[top_k.cpu().numpy()]

                top_features[i][dataset.labels_name(no_class)] = (top_k_names, top_k.cpu().numpy())
    except AttributeError:
        print "{} doesn't have any logistic layers.".format(model)

    return top_features