# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import sys
import getopt
import numpy as np
import json
from collections import OrderedDict

from pydl.nn.layers import FC
from pydl.nn.layers import NN
from pydl.training.training import Adam
from pydl import conf


def height_weight_data_split(data_file, target='height', test_smpl_per_ctgry=6):
    with open(data_file) as df:
        data_dict = json.load(df)

    category_dict = OrderedDict()
    for k, v in data_dict.items():
        try:
            category_dict[v[target+'_ctgry']].append(v['id'])
        except:
            category_dict[v[target+'_ctgry']] = list()
            category_dict[v[target+'_ctgry']].append(v['id'])

    test_subjects_dict = OrderedDict()
    test_ids = list()
    for k, v in category_dict.items():
        rnd_order = np.random.permutation(len(v))
        test_subjects_dict[k] = np.array(v)[rnd_order][:test_smpl_per_ctgry]
        test_ids.append(test_subjects_dict[k])
    test_ids = np.hstack(test_ids).tolist()

    print("Test Subject Ids per Category:")
    for i in range(len(test_subjects_dict)):

        bin = data_dict[str(test_subjects_dict[i][0])][target+'_bin']
        print("Category-%d [%.2f : %.2f]: " % (i, bin[0], bin[1]))
        print("                           ", test_subjects_dict[i])

    train_X = list()
    train_y = list()
    test_X = list()
    test_y = list()
    for k, v in data_dict.items():
        if v['id'] in test_ids:
            X = np.array(v['X'], dtype=conf.dtype)
            test_size = X.shape[0]
            test_X.append(X)
            test_y += [v[target+'_ctgry']] * test_size
        else:
            X = np.array(v['X'], dtype=conf.dtype)
            train_size = X.shape[0]
            train_X.append(X)
            train_y += [v[target+'_ctgry']] * train_size

    train_X = np.vstack(train_X)
    test_X = np.vstack(test_X)
    train_y = np.array(train_y, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)

    # Shuffle Train set
    train_size = train_X.shape[0]
    train_order = np.random.permutation(train_size)
    train_X = train_X[train_order]
    train_y = train_y[train_order]

    # Shuffle Test set
    test_size = test_X.shape[0]
    test_order = np.random.permutation(test_size)
    test_X = test_X[test_order]
    test_y = test_y[test_order]

    return train_X, test_X, train_y, test_y


def linear_classifier(X, K):
    # Linear Classifier
    l1 = FC(X, num_neurons=K, bias=True, xavier=True, activation_fn='SoftMax', name="Output-Layer")
    layers = [l1]

    return NN(X, layers)


def two_layer_nn(X, K, dropout=0.5):
    # Two-Layer Network
    l1 = FC(X, num_neurons=300, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-1")
    l2 = FC(l1, num_neurons=K, xavier=True, activation_fn='SoftMax', name="Output-Layer")
    layers = [l1, l2]

    return NN(X, layers)


def three_layer_nn(X, K, dropout=0.5):
    # Three-Layer Network
    l1 = FC(X, num_neurons=200, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-1")
    l2 = FC(l1, num_neurons=100, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-2")
    l3 = FC(l2, num_neurons=K, xavier=True, activation_fn='SoftMax', name="Output-Layer")
    layers = [l1, l2, l3]

    return NN(X, layers)


def four_layer_nn(X, K, dropout=0.5):
    # Four-Layer Network
    l1 = FC(X, num_neurons=200, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-1")
    l2 = FC(l1, num_neurons=100, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-2")
    l3 = FC(l2, num_neurons=50, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-2")
    l4 = FC(l3, num_neurons=K, xavier=True, activation_fn='SoftMax', name="Output-Layer")
    layers = [l1, l2, l3, l4]

    return NN(X, layers)


def main(argv):

    classify = 'gender'
    dims = 350
    test_samples = 6
    num_layers = 3
    dropout = 0.5
    epochs = 100
    reg_lambda = 0.1
    data_file = 'data/robot_data_file.dat'
    plot = False
    plot_title = None
    log_freq = 1

    try:
        opts, args = getopt.getopt(argv, "h pc:D:s:l:d:r:e:f:",
                                   ["plot", "classify=", "pca_dims=" "test_samples=", "num_layers=",
                                    "do=", "reg=", "epochs=", "log_freq"])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-p", "--plot"):
            plot = True
        elif opt in ("-c", "--classify"):
            classify = arg.lower()
        elif opt in ("-D", "--pca_dims"):
            dims = float(arg)
        elif opt in ("-s", "--test_samples"):
            test_samples = int(arg)
        elif opt in ("-l", "--num_layers"):
            num_layers = int(arg)
        elif opt in ("-d", "--dropout"):
            dropout = float(arg)
        elif opt in ("-r", "--reg"):
            reg_lambda = float(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-f", "--log_freq"):
            log_freq = int(arg)

    if classify in ['height', 'weight']:
        train_X, test_X, train_y, test_y = \
            height_weight_data_split(data_file, target=classify, test_smpl_per_ctgry=test_samples)

    train_size = train_X.shape[0]
    test_size = test_X.shape[0]

    X = np.vstack((train_X, test_X))
    y = np.hstack((train_y, test_y))
    K = np.max(y) + 1

    # Build a NN
    if num_layers == 1:
        nn = linear_classifier(X, K)
    elif num_layers == 2:
        nn = two_layer_nn(X, K, dropout=dropout)
    elif num_layers == 3:
        nn = three_layer_nn(X, K, dropout=dropout)
    elif num_layers == 3:
        nn = four_layer_nn(X, K, dropout=dropout)

    # Create an optimizer
    adam = Adam(nn, step_size=1e-3, beta_1=0.9,  beta_2=0.999, reg_lambda=reg_lambda,
                train_size=train_size, test_size=test_size)
    # Train NN
    adam.train(X, y, normalize='pca', dims=dims, shuffle=False, epochs=epochs, log_freq=log_freq,
               plot=plot_title)

    input("Press Enter to continue...")


if __name__ == "__main__":
    main(sys.argv[1:])
