"""
Copyright (C) 2020 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
"""

from __future__ import absolute_import, division, print_function

import pandas as pd

import tensorflow as tf

import gpflow

from sklearn.metrics import mean_absolute_error

import os
import numpy as np
import time

# file that contains functions to read dataset and run RIO variants
print(tf.__version__)

def dataset_read(dataset_name):
    if dataset_name == "yacht":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','yacht_hydrodynamics.data')
        column_names = ['Longitudinal position of the center of buoyancy','Prismatic coefficient','Length-displacement ratio','Beam-draught ratio','Length-beam ratio','Froude number','Residuary resistance']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, sep=" +")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "ENB_heating":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','ENB2012_data.xlsx')
        raw_dataset = pd.read_excel(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('Y2')
    elif dataset_name == "ENB_cooling":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','ENB2012_data.xlsx')
        raw_dataset = pd.read_excel(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('Y1')
    elif dataset_name == "airfoil_self_noise":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','airfoil_self_noise.dat')
        column_names = ['Frequency','Angle of attack','Chord length','Free-stream velocity','Suction side displacement thickness','sound pressure']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, sep="\t")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "concrete":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','Concrete_Data.xls')
        raw_dataset = pd.read_excel(dataset_path, sheet_name="Sheet1")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "winequality-red":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','winequality-red.csv')
        raw_dataset = pd.read_csv(dataset_path, sep = ';').astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "winequality-white":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','winequality-white.csv')
        raw_dataset = pd.read_csv(dataset_path, sep = ';').astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "CCPP":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','Combined_Cycle_Power_Plant.xlsx')
        raw_dataset = pd.read_excel(dataset_path, sheet_name="Sheet1")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "CASP":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','CASP.csv')
        raw_dataset = pd.read_csv(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "SuperConduct":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','SuperConduct.csv')
        raw_dataset = pd.read_csv(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "slice_localization":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','slice_localization_data.csv')
        raw_dataset = pd.read_csv(dataset_path) + 0.01
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('patientId')
    elif dataset_name == "MSD":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','YearPredictionMSD.txt')
        raw_dataset = pd.read_csv(dataset_path, sep=",", header=None)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset = dataset.astype(float)
    return dataset

def RIO_variants_running(framework_variant, kernel_type, normed_train_data, normed_test_data, train_labels, test_labels, train_NN_predictions, test_NN_predictions, M):
    train_NN_errors = train_labels - train_NN_predictions
    combined_train_data = normed_train_data.copy()
    combined_train_data['prediction'] = train_NN_predictions
    combined_test_data = normed_test_data.copy()
    combined_test_data['prediction'] = test_NN_predictions
    minibatch_size = len(normed_train_data)
    input_dimension = len(normed_train_data.columns)
    output_dimension = 1
    Z = combined_train_data.values[:M, :].copy()
    time_checkpoint1 = time.time()
    if kernel_type == "RBF+RBF":
        k = gpflow.kernels.SquaredExponential(input_dim=input_dimension, active_dims=range(input_dimension)) \
            + gpflow.kernels.SquaredExponential(input_dim=output_dimension, active_dims=range(input_dimension, input_dimension + output_dimension))
    elif kernel_type == "RBF":
        k = gpflow.kernels.SquaredExponential(input_dim=input_dimension, active_dims=range(input_dimension))
    elif kernel_type == "RBFY":
        k = gpflow.kernels.SquaredExponential(input_dim=output_dimension, active_dims=range(input_dimension, input_dimension + output_dimension))
    if framework_variant == "GP_corrected" or framework_variant == "GP_corrected_inputOnly" or framework_variant == "GP_corrected_outputOnly":
        m = gpflow.models.SVGP(combined_train_data.values, train_NN_errors.values.reshape(-1,1), kern=k, likelihood=gpflow.likelihoods.Gaussian(), Z=Z)
    elif framework_variant == "GP" or framework_variant == "GP_inputOnly" or framework_variant == "GP_outputOnly":
        m = gpflow.models.SVGP(combined_train_data.values, train_labels.values.reshape(-1,1), kern=k, likelihood=gpflow.likelihoods.Gaussian(), Z=Z)

    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    if kernel_type == "RBF+RBF":
        hyperparameter = (m.kern.kernels[0].lengthscales.value, m.kern.kernels[0].variance.value, m.kern.kernels[1].lengthscales.value, m.kern.kernels[1].variance.value, m.likelihood.variance.value)
    else:
        hyperparameter = (m.kern.lengthscales.value, m.kern.variance.value, m.likelihood.variance.value)
    mean, var = m.predict_y(combined_test_data.values)
    time_checkpoint2 = time.time()
    computation_time = time_checkpoint2-time_checkpoint1
    print("computation_time_{}: {}".format(framework_variant, time_checkpoint2-time_checkpoint1))
    if framework_variant == "GP_corrected" or framework_variant == "GP_corrected_inputOnly" or framework_variant == "GP_corrected_outputOnly":
        test_final_predictions = test_NN_predictions + mean.reshape(-1)
    elif framework_variant == "GP" or framework_variant == "GP_inputOnly" or framework_variant == "GP_outputOnly":
        test_final_predictions = mean.reshape(-1)

    MAE = mean_absolute_error(test_labels.values, test_final_predictions)
    print("test mae after {}: {}".format(framework_variant, MAE))

    num_within_interval = 0
    for i in range(len(test_labels.values)):
        if test_labels.values[i] <= test_final_predictions[i] + 1.96 * np.sqrt(var.reshape(-1)[i]) and test_labels.values[i] >= test_final_predictions[i] - 1.96 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
    PCT_within95Interval = float(num_within_interval)/len(test_labels.values)
    print("percentage of test points within 95 percent confidence interval ({}): {}".format(framework_variant, PCT_within95Interval))
    num_within_interval = 0
    for i in range(len(test_labels.values)):
        if test_labels.values[i] <= test_final_predictions[i] + 1.65 * np.sqrt(var.reshape(-1)[i]) and test_labels.values[i] >= test_final_predictions[i] - 1.65 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
    PCT_within90Interval = float(num_within_interval)/len(test_labels.values)
    print("percentage of test points within 90 percent confidence interval ({}): {}".format(framework_variant, PCT_within90Interval))
    num_within_interval = 0
    for i in range(len(test_labels.values)):
        if test_labels.values[i] <= test_final_predictions[i] + 1.0 * np.sqrt(var.reshape(-1)[i]) and test_labels.values[i] >= test_final_predictions[i] - 1.0 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
    PCT_within68Interval = float(num_within_interval)/len(test_labels.values)
    print("percentage of test points within 68 percent confidence interval ({}): {}".format(framework_variant, PCT_within68Interval))
    mean = mean.reshape(-1)
    var = var.reshape(-1)
    return MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter

