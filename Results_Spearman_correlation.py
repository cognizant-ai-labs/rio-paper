"""
Copyright (C) 2020 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from sklearn.metrics import mean_squared_error

from scipy import stats

import pickle
import os
import numpy as np
import time

# file for post-processing all the experimental results for Table 1 and Table S2
# Only run this file after generating all the experimental results

print(tf.__version__)

#change dataset_index to select which dataset to use (0-11)
dataset_index = 0

dataset_name_list = ["yacht","ENB_heating","ENB_cooling","airfoil_self_noise","concrete","winequality-red","winequality-white","CCPP","CASP","SuperConduct","slice_localization","MSD"]
minibatch_size_list = [246,614,614,1202,824,1279,3918,7654,36584,17010,42800,463715]
NN_size_list = ["64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","128+128","256+256","64+64+64+64"]
RUNS_list = [100,100,100,100,100,100,100,100,100,100,100,10]


dataset_name = dataset_name_list[dataset_index]
minibatch_size = minibatch_size_list[dataset_index]
NN_size = NN_size_list[dataset_index]
RUNS = RUNS_list[dataset_index]

print(dataset_name)

model_name = "SVGP"
M = 50
max_Z = 0.99999

RMSE_all = []
noise_variance_all = []

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

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_NN_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    computation_time_NN = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    MAE = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within95Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within90Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within68Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_mean = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_var = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    computation = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    hyperparameter = pickle.load(result_file)

num_test_point = len(Storage_test_labels[0])
print("num_test_point: {}".format(num_test_point))
RMSE_RIO = []
for i in range(len(Storage_test_labels)):
    RMSE_RIO.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_test_predictions[i] + Storage_mean[i])))
print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE_RIO).mean()))
RMSE_all.append(np.array(RMSE_RIO).mean())
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
noise_variance_all.append(np.array(noise_variance).mean())

kernel_type = "RBF"
framework_variant = "GP_corrected_inputOnly"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    MAE = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within95Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within90Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within68Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_mean = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_var = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    computation = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    hyperparameter = pickle.load(result_file)

RMSE = []
for i in range(len(Storage_test_labels)):
    RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_test_predictions[i] + Storage_mean[i])))
print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE).mean()))
RMSE_all.append(np.array(RMSE).mean())
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
noise_variance_all.append(np.array(noise_variance).mean())

kernel_type = "RBFY"
framework_variant = "GP_corrected_outputOnly"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    MAE = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within95Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within90Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within68Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_mean = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_var = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    computation = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    hyperparameter = pickle.load(result_file)

RMSE = []
for i in range(len(Storage_test_labels)):
    RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_test_predictions[i] + Storage_mean[i])))
print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE).mean()))
RMSE_all.append(np.array(RMSE).mean())
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
noise_variance_all.append(np.array(noise_variance).mean())

kernel_type = "RBF+RBF"
framework_variant = "GP"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    MAE = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within95Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within90Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within68Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_mean = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_var = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    computation = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    hyperparameter = pickle.load(result_file)

RMSE = []
for i in range(len(Storage_test_labels)):
    RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_mean[i])))
print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE).mean()))
RMSE_all.append(np.array(RMSE).mean())
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
noise_variance_all.append(np.array(noise_variance).mean())

kernel_type = "RBF"
framework_variant = "GP_inputOnly"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    MAE = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within95Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within90Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within68Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_mean = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_var = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    computation = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    hyperparameter = pickle.load(result_file)

RMSE = []
for i in range(len(Storage_test_labels)):
    RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_mean[i])))
print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE).mean()))
RMSE_all.append(np.array(RMSE).mean())
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
noise_variance_all.append(np.array(noise_variance).mean())

kernel_type = "RBFY"
framework_variant = "GP_outputOnly"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    MAE = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within95Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within90Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    PCT_within68Interval = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_mean = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    Storage_var = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    computation = pickle.load(result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'rb') as result_file:
    hyperparameter = pickle.load(result_file)

RMSE = []
for i in range(len(Storage_test_labels)):
    RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_mean[i])))
print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE).mean()))
RMSE_all.append(np.array(RMSE).mean())
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
noise_variance_all.append(np.array(noise_variance).mean())

print("dataset {}, Spearman’s Rank Correlation: {}".format(dataset_name, stats.spearmanr(RMSE_all, noise_variance_all)))
