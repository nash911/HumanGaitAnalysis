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
from os.path import join
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn')

prev_plot_time = 0

colors = ['red', 'blue', 'green', 'brown', 'pink', 'purple', 'orange', 'magenta', 'yellow', 'black',
          'cyan', 'gold', 'lavender', 'turquoise', 'violet', 'beige', 'salmon', 'olive', 'orchid',
          'crimson', 'deeppink', 'coral', 'indigo', 'slateblue', 'darkblue', 'slategrey', 'navy',
          'royalblue', 'forestgreen', 'darkorange', 'chocolate', 'sienna', 'darkred', 'rosybrown',
          'dodgerblue', 'teal', 'seagreen', 'thistle', 'firebrick', 'tomato', 'grey', 'wheat', 'tan',
          'darkslategray', 'darkgoldenrod']


def plot_graphs(ax, plot_dict, conf_intervals=True, legend=True):
    ax.clear()
    num_plots = len(plot_dict)
    ncol = int(np.ceil(num_plots / 4))

    if conf_intervals:
        for k, v in plot_dict.items():
            epochs = v['epochs']
            y_mean = np.mean(v['plot_arr'], axis=-1)
            y_std = np.std(v['plot_arr'], axis=-1)
            ci = y_std
            ax.plot(epochs, y_mean, color=v['color'], label=v['label'])
            ax.fill_between(epochs, (y_mean - ci), (y_mean + ci), color=v['color'], alpha=.1)

        ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.3), ncol=ncol, prop={'size': 15})
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)


def clac_stats(train_dict, metric_title):
    print("---------------------------------------------------------------------------------------")
    for k, v in train_dict.items():
        eval_metric = v['plot_arr']
        median_eval_metric = np.median(eval_metric[-10:, :], axis=0)
        avg_eval_metric = np.mean(median_eval_metric)
        std_eval_metric = np.std(median_eval_metric)
        med_eval_metric = np.median(median_eval_metric)
        min_eval_metric = np.min(median_eval_metric)
        max_eval_metric = np.max(median_eval_metric)

        print(v['label'], "[", median_eval_metric.size, "]")
        print((metric_title + " -- Mean: %.2f - STD: %.2f || Median: %.2f || Min: %.2f - Max: %.2f")
              % (avg_eval_metric, std_eval_metric, med_eval_metric, min_eval_metric,
                 max_eval_metric))
    print("---------------------------------------------------------------------------------------")


def extract_files(key, path='output/'):
    if path[-1] != '/':
        path += '/'

    _, _, filenames = next(os.walk(path))
    input_files = list()
    for f in filenames:
        if key in f:
            input_files.append(join(path, f))

    input_files.sort()
    return input_files


def extract_metric_title(metric):
    if 'train' in metric:
        if 'loss' in metric:
            metric = 'train_loss'
            metric_title = 'Train Loss'
        elif 'accu' in metric:
            metric = 'train_accuracy'
            metric_title = 'Train Accuracy'
    elif 'test' in metric:
        if 'loss' in metric:
            metric = 'test_loss'
            metric_title = 'Test Loss'
        elif 'accu' in metric:
            metric = 'test_accuracy'
            metric_title = 'Test Accuracy'

    return metric, metric_title


def get_input_files(input_files, path='output/'):
    for i_file in input_files:
        if '*' in i_file:
            key = i_file.split('*')[0]
            input_files += extract_files(key, path)

    for i_file in list(input_files):
        if '*' in i_file:
            input_files.remove(i_file)
            input_files.sort()

    return input_files


def usage():
    print("Usage: analyze_trained_models.py [-i | --inp_file] <file path of training_logs> \n"
          "                                 "
          "[-m | --metric] <train_loss/test_loss/train_accu/test_accu> \n"
          "                                 [-e | --epochs] <max no. of epochs to evaluate> \n"
          )


def main(argv):
    path = 'output/'
    input_files = list()
    task = list()
    metric = 'test_accuracy'
    epochs = None
    num_train_models = None

    try:
        opts, args = getopt.getopt(argv, "h p:i:m:e:n:", ["inp_path=", "inp_file=", "metric=",
                                                          "epochs=", "num_train="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-p", "--inp_path"):
            if 'output/' in arg:
                path = arg
            else:
                path += arg
        elif opt in ("-i", "--inp_file"):
            input_files.append(arg)
        elif opt in ("-m", "--metric"):
            metric = arg.lower()
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-n", "--num_train"):
            num_train_models = int(arg)

    metric, metric_title = extract_metric_title(metric)
    input_files = get_input_files(input_files, path)

    for i_file in input_files:
        if 'id' in i_file.lower() and 'ID' not in task:
            task.append('ID')
        elif 'gender' in i_file.lower() and 'Gender' not in task:
            task.append('Gender')
        elif 'height' in i_file.lower() and 'Height' not in task:
            task.append('Height')
        elif 'weight' in i_file.lower() and 'Weight' not in task:
            task.append('Weight')

    task_title = ''
    for tsk in task:
        task_title += (tsk + ' ')
    title = task_title + '- ' + metric_title + ' Plot'

    def animate(i):
        global prev_plot_time

        replot = False
        for i_file in input_files:
            input_files_mod_time = os.stat(i_file)[8]
            if (input_files_mod_time > prev_plot_time):
                prev_plot_time = os.stat(i_file)[8]
                replot = True
                break

        if replot:
            train_dict = OrderedDict()
            for ind, i_file in enumerate(input_files):
                train_inst_dict = OrderedDict()
                with open(i_file) as inp_f:
                    cross_valid_dict = json.load(inp_f)

                try:
                    train_inst_dict['epochs'] = cross_valid_dict['0']['epochs']
                    eval_metric = list()

                    for i in range(len(cross_valid_dict) - 1):
                    # for i in range(2 - 1):
                        try:
                            eval_metric.append(
                                np.reshape(np.array(cross_valid_dict[str(i)][metric]), (-1, 1)))
                        except KeyError:
                            continue

                    eval_metric = np.hstack(eval_metric)
                    train_inst_dict['plot_arr'] = eval_metric
                    train_inst_dict['color'] = colors[ind]
                    train_inst_dict['label'] = i_file.split('/')[-1].split('.')[0]
                    train_dict[ind] = train_inst_dict
                except KeyError:
                    pass

            plot_graphs(axs, train_dict, conf_intervals=True)
            clac_stats(train_dict, metric_title)

    fig, axs = plt.subplots(1, 1, sharex=True)
    fig.suptitle(title, fontsize=20)

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
