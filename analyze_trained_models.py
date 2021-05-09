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
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.lines as mlines

style.use('seaborn')

prev_plot_time = 0

colors = ['red', 'blue', 'green', 'brown', 'pink', 'purple', 'orange', 'magenta', 'tan', 'black',
          'cyan', 'gold', 'lavender', 'turquoise', 'violet', 'beige', 'salmon', 'olive',
          'crimson', 'eggplant', 'coral']


def plot_graphs(ax, plot_dict, conf_intervals=True, legend=True):
    ax.clear()
    num_plots = len(plot_dict)
    ncol = int(np.ceil(num_plots/4))

    if conf_intervals:
        for k, v in plot_dict.items():
            epochs = v['epochs']
            y_mean = np.mean(v['plot_arr'], axis=-1)
            y_std = np.std(v['plot_arr'], axis=-1)
            ci = y_std
            ax.plot(epochs, y_mean, color=v['color'], label=v['label'])
            ax.fill_between(epochs, (y_mean-ci), (y_mean+ci), color=v['color'], alpha=.1)

        ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.3), ncol=ncol, prop={'size': 15})
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)


def clac_stats(train_dict):
    print("---------------------------------------------------------------------------------------")
    for k, v in train_dict.items():
        test_accur = v['plot_arr']
        median_test_accur = np.median(test_accur[-10:,:], axis=0)
        avg_test_accur = np.mean(median_test_accur)
        std_test_accur = np.std(median_test_accur)
        med_test_accur = np.median(median_test_accur)
        min_test_accur = np.min(median_test_accur)
        max_test_accur = np.max(median_test_accur)

        print(v['label'])
        print("Test Accuracy -- Mean: %.2f - STD: %.2f || Median: %.2f || Min: %.2f - Max: %.2f" %
              (avg_test_accur, std_test_accur, med_test_accur, min_test_accur, max_test_accur))
    print("---------------------------------------------------------------------------------------")


def usage():
    print("Usage: gait_classification.py [-i | --inp_file] <file path containing training_logs> \n"
          "                              [-l | --loss] <loss plot flag> \n"
          )


def main(argv):
    inp_file = list()
    loss = False

    try:
        opts, args = getopt.getopt(argv, "h i:", ["inp_file="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-i", "--inp_file"):
            inp_file.append(arg)
        elif opt in ("-l", "--loss"):
            loss = True

    def animate(i):
        global prev_plot_time

        replot = False
        for i_file in inp_file:
            inp_file_mod_time = os.stat(i_file)[8]
            if (inp_file_mod_time > prev_plot_time):
                prev_plot_time = os.stat(i_file)[8]
                replot = True
                break

        if replot:
            train_dict = OrderedDict()
            for ind, i_file in enumerate(inp_file):
                train_inst_dict = OrderedDict()
                with open(i_file) as inp_f:
                    cross_valid_dict = json.load(inp_f)

                try:
                    train_inst_dict['epochs'] = cross_valid_dict['0']['epochs']
                    test_accur = list()
                    for i in range(len(cross_valid_dict)-1):
                        test_accur.append(np.reshape(np.array(cross_valid_dict[str(i)]['test_accuracy']),
                                                     (-1, 1)))

                    test_accur = np.hstack(test_accur)
                    train_inst_dict['plot_arr'] = test_accur
                    train_inst_dict['color'] = colors[ind]
                    train_inst_dict['label'] = i_file.split('/')[1].split('.')[0]
                    train_dict[ind] = train_inst_dict
                except:
                    pass

            plot_graphs(axs, train_dict, conf_intervals=True)
            clac_stats(train_dict)

    fig, axs = plt.subplots(1, 1, sharex=True)
    fig.suptitle('Test Accuracy Plot', fontsize=20)

    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
