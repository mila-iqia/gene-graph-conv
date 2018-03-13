import datetime
import torch
from models.models import GraphNetwork
import pickle
import os
import logging
from logger import Logger
from torch.autograd import Variable


def feature_selection(model, dataset, opt, top=100):

    """
    Save the norm of the weight of the last layer.
    :param model:
    :param dataset:
    :param opt:
    :param top:
    :return:
    """

    top_features = []  # TODO: export emb.
    if opt.attention_layer > 0:
        print "Feature selection when using attention is not implemented yet."
        return top_features

    try:
        for i, layer in enumerate(model.my_logistic_layers):
            weight = layer.weight.data

            top_features.append({})
            for no_class in range(weight.size(0)):
                nb_channel = 1 if weight[no_class].size(0) == model.nb_nodes else model.nb_channels[0]
                this_layer_feature = torch.abs(weight[no_class].view(nb_channel, model.nb_nodes)).sum(0)  # It's a logistic regression, so lets do that.

                _, top_k = torch.topk(this_layer_feature, min(top, model.nb_nodes))
                top_k_names = dataset.node_names[top_k.cpu().numpy()]

                top_features[i][dataset.labels_name(no_class)] = (top_k_names, top_k.cpu().numpy())
    except AttributeError as e:
        print e

    return top_features


def get_graph(model):

    """
    Get the graph of a model.
    :param model:
    :return:
    """

    retn = []

    if not issubclass(model.__class__, GraphNetwork):
        print "The model is not a graph convolution."
        return retn

    # Go over all convolution
    for conv_layer in model.my_convs:
        adj = conv_layer.adj
        to_keep = conv_layer.to_keep
        retn.append([adj, to_keep])

    return retn


def get_representation(model, dataset, opt):

    """
    Get the graph of a model.
    :param model:
    :return:
    """

    retn = []

    if not issubclass(model.__class__, GraphNetwork):
        print "The model is not a graph convolution."
        return retn

    # Get one representation.
    for no_b, mini in enumerate(dataset):
        inputs, targets = mini['sample'], mini['labels']
        inputs = Variable(inputs, requires_grad=False).float()

        if opt.cuda:
            inputs = inputs.cuda()

        model.eval()
        pred = model(inputs)

        # Forward pass: Compute predicted y by passing x to the model
        retn = model.get_representation()
        retn['example'] = {'input': inputs.cpu().data.numpy(), 'output': targets.cpu().numpy()}

        break

    return retn


def setup_tensorboard_log(opt):
    exp_dir = os.path.join(opt.tensorboard_dir, opt.dataset, opt.model, opt.experiment_var, opt.trial_number)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    else:
        pass  # TODO: this should be the checkpoint, load in old state

    pickle.dump(opt, open(os.path.join(exp_dir, 'options.pkl'), 'wb'))
    writer = Logger(exp_dir)
    print "We will log everything in ", exp_dir
    return writer, exp_dir

def get_state_dict(model):

    state = model.state_dict().copy()
    to_del = []

    for name, obj in state.iteritems():
        if "sparse" in name:
            to_del.append(name)
        else:
            state[name] = state[name].cpu().numpy()

    for name in to_del:
        del state[name]

    return state

def monitor_everything(model, dataset, opt, exp_dir):
    print "Saving everything in:", exp_dir
    #print "Extracting the important features..."
    #features = feature_selection(model, dataset.dataset, opt)
    #pickle.dump(features, open(os.path.join(exp_dir, 'features.pkl'), 'wb'))
    print "Saving the weights..."
    pickle.dump(get_state_dict(model), open(os.path.join(exp_dir, 'weights.pkl'), 'wb'))

    print "Extracting the graphs..."
    graphs = get_graph(model)
    pickle.dump(graphs, open(os.path.join(exp_dir, 'graphs.pkl'), 'wb'))

    print "Saving a representation..."
    rep = get_representation(model, dataset, opt)
    pickle.dump(rep, open(os.path.join(exp_dir, 'representations.pkl'), 'wb'))

    print "Saving the gradients..."
    grads = model.grads
    pickle.dump(grads, open(os.path.join(exp_dir, 'grads.pkl'), 'wb'))


    print "Done!"


def load_everything(exp_dir):
    print "Loading the data..."
    features = pickle.load(open(os.path.join(exp_dir, 'features.pkl')))
    graphs = pickle.load(open(os.path.join(exp_dir, 'graphs.pkl')))
    reps = pickle.load(open(os.path.join(exp_dir, 'representations.pkl')))
    logging.info("Done!")

    return features, graphs, reps
