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
from pydl.nn.rnn import RNN
from pydl.nn.lstm import LSTM
from pydl.nn.gru import GRU
from pydl.nn.nn import NN
from pydl.training.momentum import Momentum
from pydl.training.adam import Adam
from pydl import conf

np.random.seed(11421111)


def auto_generate_log_file_name(task, nn_type, num_layers, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if out_path[-1] != '/':
        out_path += '/'

    _, _, filenames = next(os.walk(out_path))
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
        out_path + task + '_' + nn_type + '_' + str(num_layers) + '_' + str(new_id) + '.dat'

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


def height_weight_data_split(data_file, target='height', test_smpl_per_ctgry=0.3, regression=False):
    with open(data_file) as df:
        data_dict = json.load(df)

    if regression:
        ids_list = list()
        # Collect all person-ids in the data dictionary
        for k, v in data_dict.items():
            ids_list.append(v['id'])

        # Get number of test samples
        if test_smpl_per_ctgry < 1.0:
            num_subjects = len(ids_list)
            test_smpl_per_ctgry = int(np.rint(test_smpl_per_ctgry * num_subjects))
            print("test_smpl_per_ctgry: ", test_smpl_per_ctgry)

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
            test_subjects_dict[k] = np.array(v)[rnd_order][:int(test_smpl_per_ctgry)]
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


def single_layer_recurrent_nn(X, nn_type, seq_len, num_output_neurons, out_actvn_fn, dropout=None,
                              tune=False):
    # Single Layer Recurrent Network
    if nn_type == 'RNN':
        l1 = RNN(X[0], num_neurons=50, seq_len=seq_len, xavier=True, activation_fn='Tanh',
                 architecture_type='many_to_one', dropout=dropout, tune_internal_states=tune,
                 name="RNN-1")
    elif nn_type == 'LSTM':
        l1 = LSTM(X[0], num_neurons=50, bias=1.0, seq_len=seq_len, xavier=True,
                  architecture_type='many_to_one', dropout=dropout, tune_internal_states=tune,
                  name="LSTM-1")
    elif nn_type == 'GRU':
        l1 = GRU(X[0], num_neurons=50, bias=1.0, seq_len=seq_len, xavier=True,
                 architecture_type='many_to_one', dropout=dropout, reset_pre_transform=True,
                 tune_internal_states=tune, name="GRU-1")
    l2 = FC(l1, num_neurons=num_output_neurons, xavier=True, activation_fn=out_actvn_fn,
            name="Output-Layer")
    layers = [l1, l2]

    return NN(X, layers)


def two_layer_recurrent_nn(X, nn_type, seq_len, num_output_neurons, out_actvn_fn, dropout=None,
                           tune=False):
    # Two-Layer Recurrent Network
    if nn_type == 'RNN':
        l1 = RNN(X[0], num_neurons=50, seq_len=seq_len, xavier=True, activation_fn='Tanh',
                 architecture_type='many_to_many', dropout=dropout, tune_internal_states=tune,
                 name="RNN-1")
        l2 = RNN(l1, num_neurons=50, seq_len=seq_len, xavier=True, activation_fn='Tanh',
                 architecture_type='many_to_one', dropout=dropout, tune_internal_states=tune,
                 name="RNN-2")
    elif nn_type == 'LSTM':
        l1 = LSTM(X[0], num_neurons=50, bias=True, seq_len=seq_len, xavier=True,
                  architecture_type='many_to_many', dropout=dropout, tune_internal_states=tune,
                  name="LSTM-1")
        l2 = LSTM(l1, num_neurons=50, bias=True, seq_len=seq_len, xavier=True,
                  architecture_type='many_to_one', dropout=dropout, tune_internal_states=tune,
                  name="LSTM-2")
    elif nn_type == 'GRU':
        l1 = GRU(X[0], num_neurons=50, bias=True, seq_len=seq_len, xavier=True,
                 architecture_type='many_to_many', dropout=dropout, reset_pre_transform=True,
                 tune_internal_states=tune, name="LSTM-1")
        l2 = GRU(l1, num_neurons=50, bias=True, seq_len=seq_len, xavier=True,
                 architecture_type='many_to_one', dropout=dropout, reset_pre_transform=True,
                 tune_internal_states=tune, name="LSTM-2")
    l3 = FC(l2, num_neurons=num_output_neurons, xavier=True, activation_fn=out_actvn_fn,
            name="Output-Layer")
    layers = [l1, l2, l3]

    return NN(X, layers)


def train_fold(data_file, task, nn_type, regression, binary, normalize, actv_fn, test_split,
               num_layers, step_size, dropout, reg_lambda, tune, epochs, log_freq, plot_title,
               cross_valid_dict):
    if task == 'gender':
        if test_split >= 1.0:
            sys.exit("Error: For Gender classification, test_split [-s|--test_split] must " +
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
        if not regression and test_split < 1.0:
            sys.exit("Error: For Height/Weight classification, test_split [-s|--test_split] " +
                     "must be an integer value >= 1")
        train_X, test_X, train_y, test_y, test_ids_dict = \
            height_weight_data_split(data_file, task, test_smpl_per_ctgry=test_split,
                                     regression=regression)

    train_size = train_X.shape[0]
    test_size = test_X.shape[0]

    X = np.vstack((train_X, test_X))
    y = np.hstack((train_y, test_y))

    X = X.reshape(X.shape[0], -1, 51)
    seq_len = X.shape[1]

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
            'True' if num_output_neurons == 1 and binary else 'False'

    cross_valid_dict['training_hyper_params']['num_output_neurons'] = str(num_output_neurons)
    cross_valid_dict['training_hyper_params']['activation_fn'] = out_actvation_fn

    # Build a NN
    if num_layers == 1:
        nn = single_layer_recurrent_nn(X, nn_type, seq_len, num_output_neurons, out_actvation_fn,
                                       dropout, tune)
    elif num_layers == 2:
        nn = two_layer_recurrent_nn(X, nn_type, seq_len, num_output_neurons, out_actvation_fn,
                                    dropout, tune)
    else:
        sys.exit("Error: Recurrent-NN deeper than 2-layers are not implemented. Use flag" +
                 " [-l|--num_layers] with value <= 2")

    # Create an optimizer
    adam = Adam(nn, step_size=step_size, beta_1=0.9, beta_2=0.999, reg_lambda=reg_lambda,
                train_size=train_size, test_size=test_size, regression=regression)
    # Train NN
    train_logs_dict = adam.train_recurrent(X, y, batch_size=1, epochs=epochs, log_freq=log_freq,
                                           normalize=normalize, temperature=1.0, plot=plot_title)
    train_logs_dict['test_ids'] = test_ids_dict

    return train_logs_dict


def usage():
    print("Usage: gait_classification.py [-a | --activation_fn] <output layer activation fn.>\n"
          "                              [-b | --not_binary] \n"
          "                              [-d | --dropout] <dropout percent> \n"
          "                              [-e | --epochs] <no. of training epochs> \n"
          "                              [-f | --log_freq] <log frequency epochs> \n"
          "                              [-i | --tune_int_states] \n"
          "                              [-k | --num_folds] <no. of cross validation folds> \n"
          "                              [-l | --num_layers] <no. on nn layers> \n"
          "                              [-L | --lr] <learning rate> \n"
          "                              [-n | --normalize] \n"
          "                              [-o | --out_file] <training logs file path> \n"
          "                              [-p | --out_path] <output file path> \n"
          "                              [-r | --regul] <regularization lamda> \n"
          "                              [-R | --regression] \n"
          "                              [-s | --test_split] <test split of the data> \n"
          "                              [-t | --task] <gender/id/height/weight> \n"
          "                              [-T | --nn_type] <rnn/lstm/gru> \n"
          )


def main(argv):
    task = 'gender'
    nn_type = 'RNN'
    regression = False
    actv_fn = 'softmax'
    binary = True
    normalize = False
    test_split = 0.3
    num_layers = 2
    step_size = 1e-3
    dropout = None
    epochs = 50
    reg_lambda = 0.0
    tune = False
    data_file = 'data/robot_data_file.dat'
    out_file = 'output/train_log.dat'
    out_path = 'output/'
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
        opts, args = getopt.getopt(argv, "h bRnia:t:s:l:d:L:r:e:f:k:o:T:p:",
                                   ["not_binary", "regression", "normalize", "tune_int_states",
                                    "activation_fn=", "task=", "test_split=", "num_layers=",
                                    "dropout=", "lr=", "regul=", "epochs=", "log_freq=",
                                    "num_folds=", "out_file=", "nn_type=", "out_path"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-a", "--activation_fn"):
            actv_fn = arg.lower()
        elif opt in ("-b", "--not_binary"):
            binary = False
        elif opt in ("-t", "--task"):
            task = arg.lower()
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
            normalize = True
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
        elif opt in ("-i", "--tune_int_states"):
            tune = True
        elif opt in ("-p", "--out_path"):
            out_path += arg

    if nn_type not in ['RNN', 'LSTM', 'GRU']:
        sys.exit("Error: Unknown neural network type. Use flag [-T|--nn_type] with " +
                 "<rnn/lstm/gru>")

    if normalize:
        warnings.warn("WARNING: Normalizing sequence data!\n" +
                      "To not normalize training data, remove flag [-n|--normalize]")
        input("Press Enter to continue...")

    if regression and ('gender' in task or 'id' in task):
        sys.exit("Error: Regression is only possible for 'height' and 'weight' tasks.\n" +
                 "Remove flag [-R|--regression] from the command line.")

    if out_file.lower() == 'auto':
        out_file = auto_generate_log_file_name(task, nn_type, num_layers, out_path)

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
    training_params_dict['test_split'] = test_split
    training_params_dict['num_output_neurons'] = 'None'
    training_params_dict['model'] = str(num_layers) + '-Layer ' + nn_type.upper()
    training_params_dict['normalize'] = 'Mean Normalized' if normalize else 'None'
    training_params_dict['step_size'] = step_size
    if dropout is not None and dropout < 1.0:
        training_params_dict['dropout'] = dropout
    training_params_dict['reg_lambda'] = reg_lambda
    training_params_dict['tune_internal_states'] = 'True' if tune else 'False'
    cross_valid_dict['training_hyper_params'] = training_params_dict

    with open(out_file, 'w') as fp:
        json.dump(cross_valid_dict, fp, indent=4)

    # Train k-fold cross validation models and collect training logs
    for i in range(num_folds):
        print("Training No.:", i + 1)
        cross_valid_dict[i] = train_fold(data_file, task, nn_type, regression, binary, normalize,
                                         activations[actv_fn], test_split, num_layers, step_size,
                                         dropout, reg_lambda, tune, epochs, log_freq, plot_title,
                                         cross_valid_dict)

        # Dump extracted data to file in JSON format
        with open(out_file, 'w') as fp:
            json.dump(cross_valid_dict, fp, indent=4)

    input("Press Enter to continue...")


if __name__ == "__main__":
    main(sys.argv[1:])
