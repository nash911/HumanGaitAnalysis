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
import os
import json
from collections import OrderedDict
import warnings

from pydl.nn.layers import FC
from pydl.nn.nn import NN
from pydl.training.training import Adam
from pydl import conf


def auto_generate_log_file_name(task, nn_type, num_layers):
    _, _, filenames = next(os.walk('output/'))
    task_files = list()
    for f in filenames:
        if task + '_' + nn_type in f:
            task_files.append(f)

    file_ids = [0]
    for f in task_files:
        file_ids.append(int(f.split('_')[3].split('.')[0]))

    file_ids.sort()
    new_id = file_ids[-1] + 1
    new_file_name = \
        'output/' + task + '_' + nn_type + '_' + str(num_layers) + '_' + str(new_id) + '.dat'

    return new_file_name


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
            X = np.vstack(np.array(x, dtype=conf.dtype) for x in v['X'])
            test_size = X.shape[0]
            test_X.append(X)
            test_y += [v['sex']] * test_size
        else:
            X = np.vstack(np.array(x, dtype=conf.dtype) for x in v['X'])
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
    # for k, v in data_dict.items():
    #     X = np.vstack(np.array(x, dtype=conf.dtype) for x in v['X'])
    #     sample_size = X.shape[0]
    #     order = np.random.permutation(sample_size)
    #     X = X[order]
    #     test_size = int(sample_size * test_perc)
    #     train_size = sample_size - test_size
    #     train_X.append(X[:train_size])
    #     test_X.append(X[train_size:])
    #     train_y += [v['id']] * train_size
    #     test_y += [v['id']] * test_size

    for k, v in data_dict.items():
        X = [np.array(x, dtype=conf.dtype) for x in v['X']]
        sample_size = len(X)
        order = np.random.permutation(sample_size)
        test_size = int(np.round(sample_size * test_perc))
        train_size = sample_size - test_size
        num_train_smpls = 0
        num_test_smpls = 0
        for i in range(train_size):
            train_X.append(X[order[i]])
            num_train_smpls += X[order[i]].shape[0]
        for i in range(train_size, sample_size):
            test_X.append(X[order[i]])
            num_test_smpls += X[order[i]].shape[0]
        train_y += [v['id']] * num_train_smpls
        test_y += [v['id']] * num_test_smpls

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


def height_weight_data_split(data_file, target='height', test_smpl_per_ctgry=6, regression=False):
    with open(data_file) as df:
        data_dict = json.load(df)

    if regression:
        ids_list = list()
        # Collect all person-ids in the data dictionary
        for k, v in data_dict.items():
            ids_list.append(v['id'])

        # Sort ids randomly and choose n ids as test set
        rnd_order = np.random.permutation(len(ids_list))
        test_ids = np.array(ids_list, dtype=np.int)[rnd_order][:test_smpl_per_ctgry]
        train_ids = np.array(ids_list, dtype=np.int)[rnd_order][test_smpl_per_ctgry:]
        test_ids_dict = OrderedDict()

        test_ids_dict['test_ids'] = np.sort(test_ids).tolist()
        test_ids_dict['train_ids'] = np.sort(train_ids).tolist()

        print("Test Subject Ids:\n", test_ids_dict['test_ids'])
    else:  # Classification Task
        category_dict = OrderedDict()
        for k, v in data_dict.items():
            try:
                category_dict[v[target + '_ctgry']].append(v['id'])
            except KeyError:
                category_dict[v[target + '_ctgry']] = list()
                category_dict[v[target + '_ctgry']].append(v['id'])

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
            bin = data_dict[str(test_subjects_dict[i][0])][target + '_bin']
            test_ids_dict[str(bin)] = np.sort(test_subjects_dict[i]).tolist()
            print("Category-%d [%.2f : %.2f]: " % (i, bin[0], bin[1]))
            print("                           ", test_ids_dict[str(bin)])

    train_X = list()
    train_y = list()
    test_X = list()
    test_y = list()
    for k, v in data_dict.items():
        if v['id'] in test_ids:
            X = np.vstack(np.array(x, dtype=conf.dtype) for x in v['X'])
            test_size = X.shape[0]
            test_X.append(X)
            if regression:
                test_y += [v[target]] * test_size
            else:
                test_y += [v[target + '_ctgry']] * test_size
        else:
            X = np.vstack(np.array(x, dtype=conf.dtype) for x in v['X'])
            train_size = X.shape[0]
            train_X.append(X)
            if regression:
                train_y += [v[target]] * train_size
            else:
                train_y += [v[target + '_ctgry']] * train_size

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


def train_fold(data_file, task, regression, binary, actv_fn, normalize, test_split, num_layers,
               step_size, dropout, reg_lambda, dims, epochs, log_freq, plot_title,
               cross_valid_dict):
    if task == 'gender':
        if test_split >= 1.0:
            sys.exit("Error: For Sender classification, test_split [-s|--test_split] must " +
                     "be a float value < 1.0")
        train_X, test_X, train_y, test_y, test_ids_dict = \
            gender_data_split(data_file, test_perc=test_split)
    elif 'id' in task:  # Subject Identification
        if test_split >= 1.0:
            sys.exit("Error: For Subject Identification, test_split [-s|--test_split] must " +
                     "be a float value < 1.0")
        train_X, test_X, train_y, test_y, test_ids_dict = \
            subjectId_data_split(data_file, test_perc=test_split)
    elif task in ['height', 'weight']:
        if test_split < 1.0:
            sys.exit("Error: For Height/Weight classification, test_split [-s|--test_split] " +
                     "must be an integer value >= 1")
        train_X, test_X, train_y, test_y, test_ids_dict = \
            height_weight_data_split(data_file, task, test_smpl_per_ctgry=int(test_split),
                                     regression=regression)

    train_size = train_X.shape[0]
    test_size = test_X.shape[0]

    X = np.vstack((train_X, test_X))
    y = np.hstack((train_y, test_y))

    if regression:
        num_output_neurons = 1
        out_actvation_fn = 'Linear'
    else:  # Classification problem
        K = np.max(y) + 1

        if K == 2 and binary:
            num_output_neurons = 1
        else:
            num_output_neurons = K

        if num_output_neurons == 1:
            out_actvation_fn = 'Sigmoid'
        else:
            out_actvation_fn = actv_fn

        cross_valid_dict['training_hyper_params']['binary_classification'] = \
            'True' if num_output_neurons == 1 else 'False'

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
    adam = Adam(nn, step_size=step_size, beta_1=0.9, beta_2=0.999, reg_lambda=reg_lambda,
                train_size=train_size, test_size=test_size, regression=regression)
    # Train NN
    train_logs_dict = adam.train(X, y, normalize=normalize, dims=dims, shuffle=False, epochs=epochs,
                                 log_freq=log_freq, plot=plot_title)
    train_logs_dict['test_ids'] = test_ids_dict

    return train_logs_dict


def usage():
    print("Usage: gait_classification.py [-a | --activation_fn] <output layer activation fn.>\n"
          "                              [-b | --binary] \n"
          "                              [-d | --dropout] <dropout percent> \n"
          "                              [-D | --pca_dims] <no. of PCA dims to reduce data to> \n"
          "                              [-e | --epochs] <no. of training epochs> \n"
          "                              [-f | --log_freq] <log frequency epochs> \n"
          "                              [-k | --num_folds] <no. of cross validation folds> \n"
          "                              [-l | --num_layers] <no. on nn layers> \n"
          "                              [-L | --lr] <learning rate> \n"
          "                              [-n | --normalize] <mean/pca> \n"
          "                              [-o | --out_file] <training logs file path> \n"
          "                              [-p | --plot] \n"
          "                              [-r | --regul] <regularization lamda> \n"
          "                              [-R | --regression] \n"
          "                              [-s | --test_split] <test split of the data> \n"
          "                              [-t | --task] <gender/id/height/weight> \n"
          "                              [-T | --nn_type] <fc/cnn/rnn/lstm> \n"
          )


def main(argv):
    task = 'gender'
    nn_type = 'FC'
    regression = False
    actv_fn = 'softmax'
    binary = False
    normalize = None
    dims = 350
    test_split = 0.3
    num_layers = 3
    step_size = 1e-3
    dropout = None
    epochs = 100
    reg_lambda = 0.1
    data_file = 'data/robot_data_file.dat'
    out_file = 'output/train_log.dat'
    plot = False
    plot_title = None
    log_freq = 1
    num_folds = 1

    activations = {'linear': 'Linear',
                   'sigmoid': 'Sigmoid',
                   'tanh': 'Tanh',
                   'softmax': 'SoftMax',
                   'relu': 'ReLU'
                   }

    try:
        opts, args = getopt.getopt(argv, "h bpRa:t:D:s:l:d:L:r:n:e:f:k:o:T:",
                                   ["binary", "plot", "regression", "activation_fn=", "task=",
                                    "pca_dims=" "test_split=", "num_layers=", "dropout=", "lr=",
                                    "regul=", "normalize=", "epochs=", "log_freq=", "num_folds=",
                                    "out_file=", "nn_type="])
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
        elif opt in ("-t", "--task"):
            task = arg.lower()
        elif opt in ("-D", "--pca_dims"):
            dims = float(arg)
        elif opt in ("-s", "--test_split"):
            test_split = float(arg)
        elif opt in ("-l", "--num_layers"):
            num_layers = int(arg)
        elif opt in ("-d", "--dropout"):
            dropout = float(arg)
        elif opt in ("-L", "--lr"):
            step_size = float(arg)
        elif opt in ("-r", "--regul"):
            reg_lambda = float(arg)
        elif opt in ("-R", "--regression"):
            regression = True
        elif opt in ("-n", "--normalize"):
            normalize = arg.lower()
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-f", "--log_freq"):
            log_freq = int(arg)
        elif opt in ("-k", "--num_folds"):
            num_folds = int(arg)
        elif opt in ("-o", "--out_file"):
            out_file = arg
        elif opt in ("-T", "--nn_type"):
            nn_type = arg.upper()

    if dims > 1.0:
        dims = int(dims)

    if nn_type not in ['FC', 'CNN', 'RNN', 'LSTM']:
        sys.exit("Error: Unknown neural network type. Use flag [-T|--nn_type] with " +
                 "<fc/cnn/rnn/lstm>")

    if normalize is None:
        warnings.warn("WARNING: Ignoring data normalization!\n" +
                      "To normalize/reduce training data, use flag [-n|--normalize] <pca/mean>")
        input("Press Enter to continue...")
    elif normalize not in ['pca', 'mean']:
        sys.exit("Error: Unknown normalization type. Use flag [-n|--normalize] with <pca/mean>")

    if regression and ('gender' in task or 'id' in task):
        sys.exit("Error: Regression is only possible for 'height' and 'weight' tasks.\n" +
                 "Remove flag [-R|--regression] from the command line.")

    if out_file.lower() == 'auto':
        out_file = auto_generate_log_file_name(task, nn_type, num_layers)

    print("Training log file: ", out_file)

    # Create cross validation training logs dict, and store training hyper parameters
    cross_valid_dict = OrderedDict()
    training_params_dict = OrderedDict()
    training_params_dict['data_source'] = 'robot'
    training_params_dict['task'] = 'Subject Identification' if 'id' in task else task + \
        (' regression' if regression else ' classification')
    training_params_dict['activation_fn'] = activations[actv_fn]
    if regression:
        training_params_dict['regression'] = 'True'
    else:
        training_params_dict['classification'] = 'True'
        training_params_dict['binary_classification'] = 'True' if binary else 'False'
    training_params_dict['num_output_neurons'] = 'None'
    training_params_dict['model'] = 'Linear Classifier' if num_layers == 1 else \
                                    str(num_layers) + '-Layer NN'
    training_params_dict['normalize'] = 'None' if normalize is None else normalize
    training_params_dict['step_size'] = step_size
    if dropout is not None and dropout < 1.0:
        training_params_dict['dropout'] = dropout
    training_params_dict['reg_lambda'] = reg_lambda
    if normalize == 'pca':
        training_params_dict['pca_dims'] = dims
    cross_valid_dict['training_hyper_params'] = training_params_dict

    with open(out_file, 'w') as fp:
        json.dump(cross_valid_dict, fp, indent=4)

    # Train k-fold cross validation models and collect training logs
    for i in range(num_folds):
        print("Training No.:", i + 1)
        cross_valid_dict[i] = train_fold(data_file, task, regression, binary, activations[actv_fn],
                                         normalize, test_split, num_layers, step_size, dropout,
                                         reg_lambda, dims, epochs, log_freq, plot_title,
                                         cross_valid_dict)

        # Dump extracted data to file in JSON format
        with open(out_file, 'w') as fp:
            json.dump(cross_valid_dict, fp, indent=4)

    input("Press Enter to continue...")


if __name__ == "__main__":
    main(sys.argv[1:])
