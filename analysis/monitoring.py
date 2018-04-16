import datetime
import torch
from models.models import GraphNetwork
import pickle
import os
import logging
from logger import Logger
from torch.autograd import Variable
from models.models import get_model
import hashlib


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
    :result:
    """

    result = []

    if not issubclass(model.__class__, GraphNetwork):
        print "The model is not a graph convolution."
        return result

    # Go over all convolution
    for conv_layer in model.my_convs:
        adj = conv_layer.adj
        to_keep = conv_layer.to_keep
        result.append([adj, to_keep])

    return result


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

    params = vars(opt).copy()
    del params['seed']
    params = str(params)
    this_hash = hashlib.md5(params).hexdigest()

    if opt.load_folder is None:
        exp_dir = os.path.join(opt.tensorboard_dir, opt.dataset, opt.model, opt.name, this_hash, str(opt.seed))
        print exp_dir
    else:
        exp_dir = opt.load_folder

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        pickle.dump(opt, open(os.path.join(exp_dir, 'options.pkl'), 'wb'))

    else:
        print "We will load the model"
        opt.load_folder = exp_dir
        pass  # TODO: this should be the checkpoint, load in old state

    writer = Logger(exp_dir)
    print "We will log everything in ", exp_dir
    return writer, exp_dir

def get_state_dict(model, convert_to_numpy=True):

    state = model.state_dict().copy()
    to_del = []

    for name, obj in state.iteritems():
        if "sparse" in name:
            to_del.append(name)
        else:
            if convert_to_numpy:
                state[name] = state[name].cpu().numpy()

    for name in to_del:
        del state[name]

    return state

def monitor_everything(model, dataset, opt, exp_dir):
    print "Saving everything in:", exp_dir
    print "Extracting the graphs..."
    graphs = get_graph(model)
    pickle.dump(graphs, open(os.path.join(exp_dir, 'graphs.pkl'), 'wb'))

    print "Saving a representation..."
    rep = get_representation(model, dataset, opt)
    pickle.dump(rep, open(os.path.join(exp_dir, 'representations.pkl'), 'wb'))

    #print "Saving the gradients..."
    #grads = model.grads
    #pickle.dump(grads, open(os.path.join(exp_dir, 'grads.pkl'), 'wb'))


    print "Done!"


def load_everything(exp_dir):
    print "Loading the data..."
    features = pickle.load(open(os.path.join(exp_dir, 'features.pkl')))
    graphs = pickle.load(open(os.path.join(exp_dir, 'graphs.pkl')))
    reps = pickle.load(open(os.path.join(exp_dir, 'representations.pkl')))
    logging.info("Done!")

    return features, graphs, reps

def save_checkpoint(model, optimizer, epoch, opt, exp_dir, filename='checkpoint.pth.tar'):

    state = {
            'epoch': epoch + 1,
            'state_dict': get_state_dict(model, convert_to_numpy=False),
            'optimizer' : optimizer.state_dict(),
            'opt' : opt
        }

    #import ipdb; ipdb.set_trace()

    filename = os.path.join(exp_dir, filename)
    torch.save(state, filename)

def load_checkpoint(load_folder, opt, dataset, graph, filename='checkpoint.pth.tar'):

    # Model
    model_state = None

    # Epoch
    epoch = 0

    # Optimizser
    optimizer_state = None

    # Load the states if we saved them.
    if opt.load_folder and opt.load_checkpoint:
        # Loading all the state
        filename = os.path.join(load_folder, filename)
        if os.path.isfile(filename):
            print "=> loading checkpoint '{}'".format(filename)
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']

            # Loading the options
            opt = checkpoint['opt']
            print "Loading the model with these parameters: {}".format(opt)

            # Loading the state
            model_state = checkpoint['state_dict']
            optimizer_state = checkpoint['optimizer']
            epoch = checkpoint['epoch']

            # We override some of the options between the runs, otherwise it might be a pain.
            new_opt.epoch = opt.epoch
            if str(opt.training_mode) != str(opt.training_mode):
                optimizer_state = None

            opt.training_mode = opt.training_mode
            print"=> loaded checkpoint '{}' (epoch {})".format(filename, epoch)
        else:
            print("=> no checkpoint found at '{}'".format(filename))


    # Get the network
    my_model = get_model(opt.seed,
                         opt.nb_class,
                         opt.nb_examples,
                         opt.nb_nodes,
                         opt.model,
                         opt.cuda,
                         opt.num_channel,
                         opt.num_layer,
                         opt.use_emb,
                         opt.dropout,
                         opt.training_mode,
                         opt.use_gate,
                         opt.nb_attention_head,
                         graph,
                         dataset,
                         model_state,
                         opt)

    # Get the optimizer
    optimizer = torch.optim.Adam(my_model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    print "Our model:"
    print my_model

    return my_model, optimizer, epoch, opt
