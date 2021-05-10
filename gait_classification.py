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


def gender_data_split(data_file, test_perc=0.3):
    with open(data_file) as df:
        data_dict = json.load(df)

    male_ids = list()
    female_ids = list()
    for k, v in data_dict.items():
        if v['sex'] == 0:
            male_ids.append(v['id'])
        elif v['sex'] == 1:
            female_ids.append(v['id'])

    num_males = len(male_ids)
    num_females = len(female_ids)
    male_test_size = int(np.round(num_males * test_perc))
    female_test_size = int(np.round(num_females * test_perc))

    print("\nTotal no. of male subjects: %d  --  No. of male test subjects: %d" %
          (num_males, male_test_size))
    print("Total no. of female subjects: %d  --  No. of female test subjects: %d" %
          (num_females, female_test_size))

    rnd_order_male = np.random.permutation(len(male_ids))
    test_male_ids = np.array(male_ids)[rnd_order_male][:male_test_size]

    rnd_order_female = np.random.permutation(len(female_ids))
    test_female_ids = np.array(female_ids)[rnd_order_female][:female_test_size]
    test_ids = np.hstack((test_male_ids, test_female_ids)).tolist()

    test_ids_dict = OrderedDict()
    test_ids_dict['test_male_ids'] = np.sort(np.array(test_male_ids) + 1).tolist()
    test_ids_dict['test_female_ids'] = np.sort(np.array(test_female_ids) + 1).tolist()

    print("\nTest Subject Ids:")
    print("Male  : ", test_ids_dict['test_male_ids'])
    print("Female: ", test_ids_dict['test_female_ids'])

    train_X = list()
    train_y = list()
    test_X = list()
    test_y = list()
    for k, v in data_dict.items():
        if v['id'] in test_ids:
            X = np.array(v['X'], dtype=conf.dtype)
            test_size = X.shape[0]
            test_X.append(X)
            test_y += [v['sex']] * test_size
        else:
            X = np.array(v['X'], dtype=conf.dtype)
            train_size = X.shape[0]
            train_X.append(X)
            train_y += [v['sex']] * train_size

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

    return train_X, test_X, train_y, test_y, test_ids_dict


def subjectId_data_split(data_file, test_perc=0.3):
    with open(data_file) as df:
        data_dict = json.load(df)

    train_X = list()
    train_y = list()
    test_X = list()
    test_y = list()
    for k, v in data_dict.items():
        X = np.array(v['X'], dtype=conf.dtype)
        sample_size = X.shape[0]
        order = np.random.permutation(sample_size)
        X = X[order]
        test_size = int(sample_size * test_perc)
        train_size = sample_size - test_size
        train_X.append(X[:train_size])
        test_X.append(X[train_size:])
        train_y += [v['id']] * train_size
        test_y += [v['id']] * test_size

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

    data_split_dict = OrderedDict()
    data_split_dict['train_size'] = train_size
    data_split_dict['test_size'] = test_size

    return train_X, test_X, train_y, test_y, data_split_dict


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

    test_ids_dict = OrderedDict()
    print("Test Subject Ids per Category:")
    for i in range(len(test_subjects_dict)):
        bin = data_dict[str(test_subjects_dict[i][0])][target+'_bin']
        test_ids_dict[str(bin)] = np.sort(test_subjects_dict[i]).tolist()
        print("Category-%d [%.2f : %.2f]: " % (i, bin[0], bin[1]))
        print("                           ", test_ids_dict[str(bin)])

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

    return train_X, test_X, train_y, test_y, test_ids_dict


def linear_classifier(X, num_output_neurons, out_actvn_fn):
    # Linear Classifier
    l1 = FC(X, num_neurons=num_output_neurons, bias=True, xavier=True, activation_fn=out_actvn_fn,
            name="Output-Layer")
    layers = [l1]

    return NN(X, layers)


def two_layer_nn(X, num_output_neurons, out_actvn_fn, dropout=0.5):
    # Two-Layer Network
    l1 = FC(X, num_neurons=300, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-1")
    l2 = FC(l1, num_neurons=num_output_neurons, xavier=True, activation_fn=out_actvn_fn,
            name="Output-Layer")
    layers = [l1, l2]

    return NN(X, layers)


def three_layer_nn(X, num_output_neurons, out_actvn_fn, dropout=0.5):
    # Three-Layer Network
    l1 = FC(X, num_neurons=200, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-1")
    l2 = FC(l1, num_neurons=100, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-2")
    l3 = FC(l2, num_neurons=num_output_neurons, xavier=True, activation_fn=out_actvn_fn,
            name="Output-Layer")
    layers = [l1, l2, l3]

    return NN(X, layers)


def four_layer_nn(X, num_output_neurons, out_actvn_fn, dropout=0.5):
    # Four-Layer Network
    l1 = FC(X, num_neurons=200, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-1")
    l2 = FC(l1, num_neurons=100, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-2")
    l3 = FC(l2, num_neurons=50, xavier=True, activation_fn='ReLU', batchnorm=True, dropout=dropout,
            name="HL-2")
    l4 = FC(l3, num_neurons=num_output_neurons, xavier=True, activation_fn=out_actvn_fn,
            name="Output-Layer")
    layers = [l1, l2, l3, l4]

    return NN(X, layers)


def train_fold(data_file, classify, binary, actv_fn, test_split, num_layers, dropout, reg_lambda,
               dims, epochs, log_freq, plot_title, cross_valid_dict):
    if classify == 'gender':
        if test_split >= 1.0:
            sys.exit("Error: For Sender classification, test_split [-s|--test_split] must " +
                      "be a float value < 1.0")
        train_X, test_X, train_y, test_y, test_ids_dict = \
            gender_data_split(data_file, test_perc=test_split)
    elif 'id' in classify: # Subject Identification
        if test_split >= 1.0:
            sys.exit("Error: For Subject Identification, test_split [-s|--test_split] must " +
                      "be a float value < 1.0")
        train_X, test_X, train_y, test_y, test_ids_dict = \
            subjectId_data_split(data_file, test_perc=test_split)
    elif classify in ['height', 'weight']:
        if test_split < 1.0:
            sys.exit("Error: For Height/Weight classification, test_split [-s|--test_split] " +
                      "must be an integer value >= 1")
        train_X, test_X, train_y, test_y, test_ids_dict = \
            height_weight_data_split(data_file, classify, test_smpl_per_ctgry=int(test_split))

    train_size = train_X.shape[0]
    test_size = test_X.shape[0]

    X = np.vstack((train_X, test_X))
    y = np.hstack((train_y, test_y))

    K = np.max(y) + 1

    if K == 2 and binary:
        num_output_neurons = 1
    else:
        num_output_neurons = K

    if num_output_neurons == 1:
        out_actvation_fn = 'Sigmoid'
    else:
        out_actvation_fn = actv_fn

    cross_valid_dict['training_hyper_params']['num_output_neurons'] = str(num_output_neurons)
    cross_valid_dict['training_hyper_params']['activation_fn'] = out_actvation_fn

    # Build a NN
    if num_layers == 1:
        nn = linear_classifier(X, num_output_neurons, out_actvn_fn=out_actvation_fn)
    elif num_layers == 2:
        nn = two_layer_nn(X, num_output_neurons, out_actvn_fn=out_actvation_fn, dropout=dropout)
    elif num_layers == 3:
        nn = three_layer_nn(X, num_output_neurons, out_actvn_fn=out_actvation_fn, dropout=dropout)
    elif num_layers == 4:
        nn = four_layer_nn(X, num_output_neurons, out_actvn_fn=out_actvation_fn, dropout=dropout)
    else:
        sys.exit("Error: NN deeper than 4-layers are not implemented. Use flag [-l|--num_layers]" +
                 " with value <= 4")

    # Create an optimizer
    adam = Adam(nn, step_size=1e-3, beta_1=0.9,  beta_2=0.999, reg_lambda=reg_lambda,
                train_size=train_size, test_size=test_size)
    # Train NN
    train_logs_dict = adam.train(X, y, normalize='pca', dims=dims, shuffle=False, epochs=epochs,
                                 log_freq=log_freq, plot=plot_title)
    train_logs_dict['test_ids'] = test_ids_dict

    return train_logs_dict


def usage():
    print("Usage: gait_classification.py [-a | --activation_fn] <output layer activation fn.>\n"
          "                              [-b | --binary] \n"
          "                              [-c | --classify] <gender/id/height/weight> \n"
          "                              [-d | --dropout] <dropout percent> \n"
          "                              [-D | --pca_dims] <no. of PCA dimensions to reduce data to> \n"
          "                              [-e | --epochs] <no. of training epochs> \n"
          "                              [-f | --log_freq] <log frequency epochs> \n"
          "                              [-k | --num_folds] <no. of cross validation folds> \n"
          "                              [-l | --num_layers] <no. on nn layers> \n"
          "                              [-o | --out_file] <training logs file path> \n"
          "                              [-p | --plot] \n"
          "                              [-r | --reg] <regularization lamda> \n"
          "                              [-s | --test_split] <test split of the data> \n"
          )


def main(argv):
    classify = 'gender'
    actv_fn = 'softmax'
    binary = False
    dims = 350
    test_split = 0.3
    num_layers = 3
    dropout = 0.5
    epochs = 100
    reg_lambda = 0.1
    data_file = 'data/robot_data_file.dat'
    out_file = 'output/train_log.dat'
    plot = False
    plot_title = None
    log_freq = 1
    num_folds = 1

    activations = {'linear' : 'Linear',
                   'sigmoid' : 'Sigmoid',
                   'tanh' : 'Tanh',
                   'softmax' : 'SoftMax',
                   'relu' : 'ReLU'
                  }

    try:
        opts, args = getopt.getopt(argv, "h bpa:c:D:s:l:d:r:e:f:k:o:",
                                   ["binary", "plot", "activation_fn=", "classify=", "pca_dims="
                                    "test_split=", "num_layers=", "dropout=", "reg=", "epochs=",
                                    "log_freq=", "num_folds=", "out_file="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-a", "--activation_fn"):
            actv_fn = arg.lower()
        elif opt in ("-b", "--binary"):
            binary = True
        elif opt in ("-p", "--plot"):
            plot = True
        elif opt in ("-c", "--classify"):
            classify = arg.lower()
        elif opt in ("-D", "--pca_dims"):
            dims = float(arg)
        elif opt in ("-s", "--test_split"):
            test_split = float(arg)
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
        elif opt in ("-k", "--num_folds"):
            num_folds = int(arg)
        elif opt in ("-o", "--out_file"):
            out_file = arg

    if dims > 1.0:
        dims = int(dims)

    # Create cross validation training logs dict, and store training hyper parameters
    cross_valid_dict = OrderedDict()
    training_params_dict = OrderedDict()
    training_params_dict['data_source'] = 'robot'
    training_params_dict['classify'] = 'Subject Identification' if 'id' in classify else \
                                       classify + ' classification'
    training_params_dict['activation_fn'] = activations[actv_fn]
    training_params_dict['binary_classification'] = 'True' if binary else 'False'
    training_params_dict['num_output_neurons'] = 'None'
    training_params_dict['model'] = 'Linear Classifier' if num_layers == 1 else \
                                    str(num_layers) + '-Layer NN'
    training_params_dict['dropout'] = dropout
    training_params_dict['reg_lambda'] = reg_lambda
    training_params_dict['pca_dims'] = dims
    cross_valid_dict['training_hyper_params'] = training_params_dict

    with open(out_file, 'w') as fp:
        json.dump(cross_valid_dict, fp, indent=4)

    # Train k-fold cross validation models and collect training logs
    for i in range(num_folds):
        print("Training No.:", i+1)
        cross_valid_dict[i] = train_fold(data_file, classify, binary, activations[actv_fn],
                                         test_split, num_layers, dropout, reg_lambda, dims, epochs,
                                         log_freq, plot_title, cross_valid_dict)

        # Dump extracted data to file in JSON format
        with open(out_file, 'w') as fp:
            json.dump(cross_valid_dict, fp, indent=4)

    input("Press Enter to continue...")


if __name__ == "__main__":
    main(sys.argv[1:])
