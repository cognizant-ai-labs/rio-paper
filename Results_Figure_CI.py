"""
Copyright (C) 2020 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

import pickle
import os

# File for plotting figures in Figure 4, Figure S1 and Figure S2
# Only run this file after generating all the experimental results

def draw_plot(data, edge_color, fill_color):
    bp = plt.boxplot(data, sym=edge_color, patch_artist=True, widths = 0.5)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color, alpha=0.5)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color, alpha=0.5)

dataset_name_list = ["yacht","ENB_heating","ENB_cooling","airfoil_self_noise","concrete","winequality-red","winequality-white","CCPP","CASP","SuperConduct","slice_localization","MSD"]
title_name_list = ["yacht","ENB/h","ENB/c","airfoil","CCS","wine/r","wine/w","CCPP","protein","SC","CT","MSD"]
minibatch_size_list = [246,614,614,1202,824,1279,3918,7654,36584,17010,42800,463715]
NN_size_list = ["64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","128+128","256+256","64+64+64+64"]
RUNS_list = [100,100,100,100,100,100,100,100,100,100,100,10]

model_name = "SVGP"
optimizer_name = "LBFGSB"
M = 50
alpha_value = 0.5

for dataset_index in range(0,len(dataset_name_list)):
    dataset_name = dataset_name_list[dataset_index]
    title_name = title_name_list[dataset_index]
    minibatch_size = minibatch_size_list[dataset_index]
    NN_size = NN_size_list[dataset_index]
    RUNS = RUNS_list[dataset_index]
    kernel_type = "RBF+RBF"
    optimizer_name = "LBFGSB"
    framework_variant = "GP_corrected"
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within95Interval_GPcorrected = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within90Interval_GPcorrected = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within68Interval_GPcorrected = pickle.load(result_file)

    kernel_type = "RBF"
    framework_variant = "GP_corrected_inputOnly"
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within95Interval_GPcorrected_inputOnly = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within90Interval_GPcorrected_inputOnly = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within68Interval_GPcorrected_inputOnly = pickle.load(result_file)

    kernel_type = "RBFY"
    framework_variant = "GP_corrected_outputOnly"
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within95Interval_GPcorrected_outputOnly = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within90Interval_GPcorrected_outputOnly = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within68Interval_GPcorrected_outputOnly = pickle.load(result_file)

    kernel_type = "RBF+RBF"
    framework_variant = "GP"
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within95Interval_GP = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within90Interval_GP = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within68Interval_GP = pickle.load(result_file)

    kernel_type = "RBF"
    framework_variant = "GP_inputOnly"
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within95Interval_GP_inputOnly = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within90Interval_GP_inputOnly = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within68Interval_GP_inputOnly = pickle.load(result_file)

    kernel_type = "RBFY"
    framework_variant = "GP_outputOnly"
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within95Interval_GP_outputOnly = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within90Interval_GP_outputOnly = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within68Interval_GP_outputOnly = pickle.load(result_file)

    PCT_within95Interval = []
    PCT_within95Interval.append(PCT_within95Interval_GPcorrected)
    PCT_within95Interval.append(PCT_within95Interval_GPcorrected_inputOnly)
    PCT_within95Interval.append(PCT_within95Interval_GPcorrected_outputOnly)
    PCT_within95Interval.append(PCT_within95Interval_GP_outputOnly)
    PCT_within95Interval.append(PCT_within95Interval_GP)
    PCT_within95Interval.append(PCT_within95Interval_GP_inputOnly)

    f = plt.figure()
    plt.title("{}: percentage of test points within 95% CI".format(title_name))
    plt.boxplot(PCT_within95Interval)
    plt.ylabel('percentage of test points within 95% CI')
    plt.xlabel('algorithm')
    plt.xticks(range(1,7),('RIO','R+I','R+O','Y+O','Y+IO','SVGP'))
    plt.yticks(list(plt.yticks()[0]) + [0.95])
    _ = plt.plot([-100, 100], [0.95, 0.95], 'r--', alpha=alpha_value)
    plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within95Interval_comparison_{}.pdf'.format(dataset_name))
    f.savefig(plot_file_name, bbox_inches='tight')

    PCT_within90Interval = []
    PCT_within90Interval.append(PCT_within90Interval_GPcorrected)
    PCT_within90Interval.append(PCT_within90Interval_GPcorrected_inputOnly)
    PCT_within90Interval.append(PCT_within90Interval_GPcorrected_outputOnly)
    PCT_within90Interval.append(PCT_within90Interval_GP_outputOnly)
    PCT_within90Interval.append(PCT_within90Interval_GP)
    PCT_within90Interval.append(PCT_within90Interval_GP_inputOnly)

    f = plt.figure()
    plt.title("{}: percentage of test points within 90% CI".format(title_name))
    plt.boxplot(PCT_within90Interval)
    plt.ylabel('percentage of test points within 90% CI')
    plt.xlabel('algorithm')
    plt.xticks(range(1,7),('RIO','R+I','R+O','Y+O','Y+IO','SVGP'))
    plt.yticks(list(plt.yticks()[0]) + [0.90])
    _ = plt.plot([-100, 100], [0.90, 0.90], 'r--', alpha=alpha_value)
    plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within90Interval_comparison_{}.pdf'.format(dataset_name))
    f.savefig(plot_file_name, bbox_inches='tight')

    PCT_within68Interval = []
    PCT_within68Interval.append(PCT_within68Interval_GPcorrected)
    PCT_within68Interval.append(PCT_within68Interval_GPcorrected_inputOnly)
    PCT_within68Interval.append(PCT_within68Interval_GPcorrected_outputOnly)
    PCT_within68Interval.append(PCT_within68Interval_GP_outputOnly)
    PCT_within68Interval.append(PCT_within68Interval_GP)
    PCT_within68Interval.append(PCT_within68Interval_GP_inputOnly)

    f = plt.figure()
    plt.title("{}: percentage of test points within 68% CI".format(title_name))
    plt.boxplot(PCT_within68Interval)
    plt.ylabel('percentage of test points within 68% CI')
    plt.xlabel('algorithm')
    plt.xticks(range(1,7),('RIO','R+I','R+O','Y+O','Y+IO','SVGP'))
    plt.yticks(list(plt.yticks()[0]) + [0.68])
    _ = plt.plot([-100, 100], [0.68, 0.68], 'r--', alpha=alpha_value)
    plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within68Interval_comparison_{}.pdf'.format(dataset_name))
    f.savefig(plot_file_name, bbox_inches='tight')

    plt.show()



