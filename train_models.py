import sys
import os
import shutil
import itertools
import subprocess
from subprocess import Popen, PIPE
from datetime import datetime


def work(cmd):
    return subprocess.call(cmd, shell=True)


def main(argv):

    num_folds = 20
    epochs = 50

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d-%H")
    out_path = str('output/' + date_time)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        shutil.rmtree(out_path)
        os.makedirs(out_path)

    # Combinatorial Training Cases
    # ----------------------------
    task = ['gender', 'id', 'height', 'weight']
    nn_type = ['FC', 'RNN', 'LSTM']
    num_layers = [1, 2, 3, 4]
    num_neurons = [5, 15, 30]
    optimizer = ['adam', 'momentum', 'rmsprop']
    learning_rate = [1e-3]
    normalize = ['mean', 'minmax', 'pca']
    pca_dims = [350]
    dropout = [1.0, 0.9]
    regularization = [0, 0.1]
    hidden_activation_fn = ['Tanh', 'Sigmoid', 'ReLU']
    test_split = [0.25, 0.3]
    tune_internal_states = [True, False]
    command_list = list()

    for tsk, n_type, n_layer, opt, lr, norm, dout, reg, t_split in \
        list(itertools.product(task, nn_type, num_layers, optimizer, learning_rate, normalize,
                               dropout, regularization, test_split)):
        if n_type != 'FC':
            continue

        if norm not in ['mean', 'pca']:
            continue

        if n_layer == 1 and dout < 1.0:
            continue

        train_file = 'gait_classification_fc.py'
        cmd = 'python ' + train_file + ' --task ' + tsk + ' --num_layers ' + str(n_layer) + \
              ' --optimizer ' + opt + ' --lr ' + str(lr) + ' --dropout ' + str(dout) + \
              ' --regul ' + str(reg) + ' --normalize ' + norm + ' --test_split ' + str(t_split) + \
              ' --num_folds ' + str(num_folds) + ' --epochs ' + str(epochs) + ' --out_file auto' + \
              ' --out_path ' + out_path + ' --presplit_data'

        if tsk in ['height', 'weight']:
            cmd += ' --regression'

        if norm == 'pca':
            for dim in pca_dims:
                pca_dim_cmd = cmd + ' --pca_dims ' + str(dim)
                command_list.append(pca_dim_cmd)
        else:
            command_list.append(cmd)

    for tsk, n_type, n_layer, n_neurons, opt, lr, norm, dout, reg, actv, t_split, tune in \
        list(itertools.product(task, nn_type, num_layers, num_neurons, optimizer, learning_rate,
                               normalize, dropout, regularization, hidden_activation_fn, test_split,
                               tune_internal_states)):
        if n_type not in ['RNN', 'LSTM', 'GRU']:
            continue

        if norm not in ['mean', 'minmax']:
            continue

        if n_layer >= 3:
            continue

        if n_layer == 1 and dout < 1.0:
            continue

        if n_type in ['LSTM', 'GRU'] and actv != 'Tanh':
            continue

        train_file = 'gait_classification_rnn.py'
        cmd = 'python ' + train_file + ' --task ' + tsk + ' --nn_type ' + n_type + \
              ' --num_layers ' + str(n_layer) + ' --num_neurons ' + str(n_neurons) + \
              ' --hidden_activation ' + str(actv) + ' --optimizer ' + opt + ' --lr ' + str(lr) + \
              ' --dropout ' + str(dout) + ' --regul ' + str(reg) + ' --test_split ' + \
              str(t_split) + ' --num_folds ' + str(num_folds) + ' --epochs ' + str(epochs) + \
              ' --out_file auto' + ' --out_path ' + out_path + ' --presplit_data' + \
              ' --normalize ' + norm

        if tsk in ['height', 'weight']:
            cmd += ' --regression'

        if tune:
            cmd += ' --tune_int_states'

        command_list.append(cmd)

    num_commands = len(command_list)
    for id, cmd in enumerate(command_list):
        cmd += " --plot " + str('%d/%d' % (id + 1, num_commands))
        print("\nTrain-ID: ", id + 1)
        print("Command: ", cmd, '\n')

        p = Popen([cmd], stdin=PIPE, shell=True)
        p.communicate(input=b'\n')


if __name__ == "__main__":
    main(sys.argv[1:])
