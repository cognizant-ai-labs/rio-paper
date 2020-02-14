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
import sys
import numpy as np
import time
import math

# file for post-processing all the experimental results for Table 1 and Table S2
# Only run this file after generating all the experimental results

print(tf.__version__)

#change dataset_index to select which dataset to use (0-11)
dataset_index = 11

dataset_name_list = ["yacht","ENB_heating","ENB_cooling","airfoil_self_noise","concrete","winequality-red","winequality-white","CCPP","CASP","SuperConduct","slice_localization","MSD"]
minibatch_size_list = [246,614,614,1202,824,1279,3918,7654,36584,17010,42800,463715]
NN_size_list = ["64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","128+128","256+256","64+64+64+64"]
RUNS_list = [100,100,100,100,100,100,100,100,100,100,100,10]

dataset_name = dataset_name_list[dataset_index]
minibatch_size = minibatch_size_list[dataset_index]
NN_size = NN_size_list[dataset_index]
RUNS = RUNS_list[dataset_index]

def norm_pdf(x, mean, var):
    return np.exp(-(x-mean)**2/(2.0*var))/np.sqrt(2*np.pi*var)

model_name = "SVGP"
M = 50
max_Z = 0.99999

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
RMSE = []
for i in range(len(Storage_test_labels)):
    RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_test_predictions[i])))
print("{} NN RMSE mean: {}".format(framework_variant, np.array(RMSE).mean()))
print("{} NN RMSE std: {}".format(framework_variant, np.array(RMSE).std()))
print("{} NN computation time: {}".format(framework_variant, np.array(computation_time_NN).mean()))

RMSE_RIO = []
for i in range(len(Storage_test_labels)):
    RMSE_RIO.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_test_predictions[i] + Storage_mean[i])))
print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE_RIO).mean()))
print("{} RMSE std: {}".format(framework_variant, np.array(RMSE_RIO).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(RMSE, RMSE_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(RMSE, RMSE_RIO)))
print("{} computation time: {}".format(framework_variant, np.array(computation).mean()))
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
#print("{} hyperparameters: {}".format(framework_variant, np.mean(np.array(hyperparameter), axis=0)))
pdf_all_RIO = []
for run in range(len(Storage_test_labels)):
    pdf_tmp = []
    for i in range(len(Storage_test_labels[run])):
        pdf_tmp.append(-math.log(norm_pdf(Storage_test_labels[run][i], Storage_test_predictions[run][i] + Storage_mean[run][i], Storage_var[run][i])+sys.float_info.epsilon))
    pdf_all_RIO.append(np.array(pdf_tmp).mean())
print("{} NLPD mean: {}".format(framework_variant, np.array(pdf_all_RIO).mean()))
print("{} NLPD std: {}".format(framework_variant, np.array(pdf_all_RIO).std()))

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
print("{} RMSE std: {}".format(framework_variant, np.array(RMSE).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(RMSE, RMSE_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(RMSE, RMSE_RIO)))
print("{} computation time: {}".format(framework_variant, np.array(computation).mean()))
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
#print("{} hyperparameters: {}".format(framework_variant, np.mean(np.array(hyperparameter), axis=0)))
pdf_all = []
for run in range(len(Storage_test_labels)):
    pdf_tmp = []
    for i in range(len(Storage_test_labels[run])):
        pdf_tmp.append(-math.log(norm_pdf(Storage_test_labels[run][i], Storage_test_predictions[run][i] + Storage_mean[run][i], Storage_var[run][i])+sys.float_info.epsilon))
    pdf_all.append(np.array(pdf_tmp).mean())
print("{} NLPD mean: {}".format(framework_variant, np.array(pdf_all).mean()))
print("{} NLPD std: {}".format(framework_variant, np.array(pdf_all).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(pdf_all, pdf_all_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(pdf_all, pdf_all_RIO)))

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
print("{} RMSE std: {}".format(framework_variant, np.array(RMSE).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(RMSE, RMSE_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(RMSE, RMSE_RIO)))
print("{} computation time: {}".format(framework_variant, np.array(computation).mean()))
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
#print("{} hyperparameters: {}".format(framework_variant, np.mean(np.array(hyperparameter), axis=0)))
pdf_all = []
for run in range(len(Storage_test_labels)):
    pdf_tmp = []
    for i in range(len(Storage_test_labels[run])):
        pdf_tmp.append(-math.log(norm_pdf(Storage_test_labels[run][i], Storage_test_predictions[run][i] + Storage_mean[run][i], Storage_var[run][i])+sys.float_info.epsilon))
    pdf_all.append(np.array(pdf_tmp).mean())
print("{} NLPD mean: {}".format(framework_variant, np.array(pdf_all).mean()))
print("{} NLPD std: {}".format(framework_variant, np.array(pdf_all).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(pdf_all, pdf_all_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(pdf_all, pdf_all_RIO)))

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
print("{} RMSE std: {}".format(framework_variant, np.array(RMSE).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(RMSE, RMSE_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(RMSE, RMSE_RIO)))
print("{} computation time: {}".format(framework_variant, np.array(computation).mean()))
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
#print("{} hyperparameters: {}".format(framework_variant, np.mean(np.array(hyperparameter), axis=0)))
pdf_all = []
for run in range(len(Storage_test_labels)):
    pdf_tmp = []
    for i in range(len(Storage_test_labels[run])):
        pdf_tmp.append(-math.log(norm_pdf(Storage_test_labels[run][i], Storage_mean[run][i], Storage_var[run][i])+sys.float_info.epsilon))
    pdf_all.append(np.array(pdf_tmp).mean())
print("{} NLPD mean: {}".format(framework_variant, np.array(pdf_all).mean()))
print("{} NLPD std: {}".format(framework_variant, np.array(pdf_all).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(pdf_all, pdf_all_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(pdf_all, pdf_all_RIO)))

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
print("{} RMSE std: {}".format(framework_variant, np.array(RMSE).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(RMSE, RMSE_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(RMSE, RMSE_RIO)))
print("{} computation time: {}".format(framework_variant, np.array(computation).mean()))
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
#print("{} hyperparameters: {}".format(framework_variant, np.mean(np.array(hyperparameter), axis=0)))
pdf_all = []
for run in range(len(Storage_test_labels)):
    pdf_tmp = []
    for i in range(len(Storage_test_labels[run])):
        pdf_tmp.append(-math.log(norm_pdf(Storage_test_labels[run][i], Storage_mean[run][i], Storage_var[run][i])+sys.float_info.epsilon))
    pdf_all.append(np.array(pdf_tmp).mean())
print("{} NLPD mean: {}".format(framework_variant, np.array(pdf_all).mean()))
print("{} NLPD std: {}".format(framework_variant, np.array(pdf_all).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(pdf_all, pdf_all_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(pdf_all, pdf_all_RIO)))

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
print("{} RMSE std: {}".format(framework_variant, np.array(RMSE).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(RMSE, RMSE_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(RMSE, RMSE_RIO)))
print("{} computation time: {}".format(framework_variant, np.array(computation).mean()))
noise_variance = []
for i in range(len(hyperparameter)):
    noise_variance.append(hyperparameter[i][-1])
print("{} noise variance: {}".format(framework_variant, np.array(noise_variance).mean()))
#print("{} hyperparameters: {}".format(framework_variant, np.mean(np.array(hyperparameter), axis=0)))
pdf_all = []
for run in range(len(Storage_test_labels)):
    pdf_tmp = []
    for i in range(len(Storage_test_labels[run])):
        pdf_tmp.append(-math.log(norm_pdf(Storage_test_labels[run][i], Storage_mean[run][i], Storage_var[run][i])+sys.float_info.epsilon))
    pdf_all.append(np.array(pdf_tmp).mean())
print("{} NLPD mean: {}".format(framework_variant, np.array(pdf_all).mean()))
print("{} NLPD std: {}".format(framework_variant, np.array(pdf_all).std()))
print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(pdf_all, pdf_all_RIO)))
print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(pdf_all, pdf_all_RIO)))

if dataset_index < 4:
    kernel_type = "NNGP"
    framework_variant = "NNGP"
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
    RMSE = []
    for i in range(len(Storage_test_labels)):
        RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_mean[i])))
    print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE).mean()))
    print("{} RMSE std: {}".format(framework_variant, np.array(RMSE).std()))
    print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(RMSE, RMSE_RIO)))
    print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(RMSE, RMSE_RIO)))
    print("{} computation time: {}".format(framework_variant, np.array(computation).mean()*900.0))
    pdf_all = []
    for run in range(len(Storage_test_labels)):
        pdf_tmp = []
        for i in range(len(Storage_test_labels[run])):
            pdf_tmp.append(-math.log(norm_pdf(Storage_test_labels[run][i], Storage_mean[run][i], Storage_var[run][i])+sys.float_info.epsilon))
        pdf_all.append(np.array(pdf_tmp).mean())
    print("{} NLPD mean: {}".format(framework_variant, np.array(pdf_all).mean()))
    print("{} NLPD std: {}".format(framework_variant, np.array(pdf_all).std()))
    print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(pdf_all, pdf_all_RIO)))
    print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(pdf_all, pdf_all_RIO)))


    TRAINING_ITERATIONS = 2000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
    PLOT_AFTER = 50#@param {type:"number"}
    HIDDEN_SIZE = 64 #@param {type:"number"}
    MODEL_TYPE = 'ANP' #@param ['NP','ANP']
    ATTENTION_TYPE = 'multihead' #@param ['uniform','laplace','dot_product','multihead']
    random_kernel_parameters=True #@param {type:"boolean"}
    context_ratio = 0.8

    latent_encoder_output_sizes = [HIDDEN_SIZE]*4
    num_latents = HIDDEN_SIZE
    deterministic_encoder_output_sizes= [HIDDEN_SIZE]*4
    decoder_output_sizes = [HIDDEN_SIZE]*2 + [2]
    use_deterministic_path = True
    kernel_type = "RBF+RBF"
    optimizer_name = "Adam"
    framework_variant = "ANP"

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_Iteration{}_Gap{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, TRAINING_ITERATIONS, PLOT_AFTER, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        MAE = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_Iteration{}_Gap{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, TRAINING_ITERATIONS, PLOT_AFTER, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within95Interval = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_Iteration{}_Gap{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, TRAINING_ITERATIONS, PLOT_AFTER, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within90Interval = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_Iteration{}_Gap{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, TRAINING_ITERATIONS, PLOT_AFTER, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        PCT_within68Interval = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_Iteration{}_Gap{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, TRAINING_ITERATIONS, PLOT_AFTER, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        Storage_mean = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_Iteration{}_Gap{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, TRAINING_ITERATIONS, PLOT_AFTER, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        Storage_var = pickle.load(result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_Iteration{}_Gap{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, TRAINING_ITERATIONS, PLOT_AFTER, minibatch_size, optimizer_name, RUNS))
    with open(result_file_name, 'rb') as result_file:
        time = pickle.load(result_file)

    gap_num = 39
    RMSE = []
    for i in range(len(Storage_test_labels)):
        RMSE.append(np.sqrt(mean_squared_error(Storage_test_labels[i], Storage_mean[i][gap_num])))
    print("{} RMSE mean: {}".format(framework_variant, np.array(RMSE).mean()))
    print("{} RMSE std: {}".format(framework_variant, np.array(RMSE).std()))
    print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(RMSE, RMSE_RIO)))
    print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(RMSE, RMSE_RIO)))
    print("{} computation time: {}".format(framework_variant, np.array(time).mean(axis=0)[gap_num]))
    pdf_all = []
    for run in range(len(Storage_test_labels)):
        pdf_tmp = []
        for i in range(len(Storage_test_labels[run])):
            pdf_tmp.append(-math.log(norm_pdf(Storage_test_labels[run][i], Storage_mean[run][gap_num][i], Storage_var[run][gap_num][i])+sys.float_info.epsilon))
        pdf_all.append(np.array(pdf_tmp).mean())
    print("{} NLPD mean: {}".format(framework_variant, np.array(pdf_all).mean()))
    print("{} NLPD std: {}".format(framework_variant, np.array(pdf_all).std()))
    print("{} ttest_rel score: {}".format(framework_variant, stats.ttest_rel(pdf_all, pdf_all_RIO)))
    print("{} wilcoxon score: {}".format(framework_variant, stats.wilcoxon(pdf_all, pdf_all_RIO)))
