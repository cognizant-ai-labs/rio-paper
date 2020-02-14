"""
Copyright (C) 2020 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import pickle
import os
import numpy as np

# File for plotting figures in Figure 3
# Only run this file after generating all the experimental results

dataset_name_list = ["yacht","ENB_heating","ENB_cooling","airfoil_self_noise","concrete","winequality-red","winequality-white","CCPP","CASP","SuperConduct","slice_localization","MSD"]
title_name_list = ["yacht","ENB/h","ENB/c","airfoil","CCS","wine/r","wine/w","CCPP","protein","SC","CT","MSD"]
minibatch_size_list = [246,614,614,1202,824,1279,3918,7654,36584,17010,42800,463715]
NN_size_list = ["64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","128+128","256+256","64+64+64+64"]
RUNS_list = [100,100,100,100,100,100,100,100,100,100,100,10]
alpha_value = 0.5

model_name = "SVGP"
M = 50

for k in range(len(dataset_name_list)):
    dataset_name = dataset_name_list[k]
    title_name = title_name_list[k]
    minibatch_size = minibatch_size_list[k]
    NN_size = NN_size_list[k]
    RUNS = RUNS_list[k]
    kernel_type = "RBF+RBF"
    optimizer_name = "LBFGSB"
    framework_variant = "GP_corrected"

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_NN_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        MAE_original = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_test_labels_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        Storage_test_labels = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_test_NN_predictions_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        Storage_test_predictions = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        MAE = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        Storage_mean = pickle.load(result_file)

    num_test_point = len(Storage_test_labels[0])
    print("num_test_point: {}".format(num_test_point))
    RMSE_NN = []
    for i in range(len(Storage_test_labels)):
        RMSE_NN.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_test_predictions[i])))

    RMSE_RIO = []
    for i in range(len(Storage_test_labels)):
        RMSE_RIO.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_test_predictions[i] + Storage_mean[i])))

    f = plt.figure()
    plt.title(title_name)
    plt.scatter(RMSE_NN, RMSE_RIO, label="RIO", alpha=alpha_value)

    kernel_type = "RBF"
    framework_variant = "GP_inputOnly"
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        Storage_mean = pickle.load(result_file)

    RMSE = []
    for i in range(len(Storage_test_labels)):
        RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_mean[i])))
    if k!=10:
        plt.scatter(RMSE_NN, RMSE, label="SVGP", alpha=alpha_value)
    plt.xlabel('NN RMSE')
    plt.ylabel('RIO/SVGP RMSE')
    bottom, top = plt.ylim()
    left, right = plt.xlim()
    _ = plt.plot([-100, 100], [-100, 100])
    if k == 7:
        top = 10
        bottom = 3.6
    plt.ylim((bottom, top))
    plt.xlim((left, right))

    legend = plt.legend()
    plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','RMSE_comparison_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
    f.savefig(plot_file_name, bbox_inches='tight')

plt.show()
