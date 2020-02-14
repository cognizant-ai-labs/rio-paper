"""
Copyright (C) 2020 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle
import os
import time
from util import dataset_read, RIO_variants_running

#main file to run tests for all the RIO variants on all the datasets

print(tf.__version__)

#change dataset_index to select which dataset to use (0-11)
dataset_index = 0
model_name = "SVGP"
#number of Epochs for NN training
EPOCHS = 1000
#number of inducing points for SVGP
M = 50

dataset_name_list = ["yacht","ENB_heating","ENB_cooling","airfoil_self_noise","concrete","winequality-red","winequality-white","CCPP","CASP","SuperConduct","slice_localization","MSD"]
label_name_list = ["Residuary resistance", "Y1", "Y2", "sound pressure", "Mpa", "quality", "quality", "PE", "RMSD", "critical_temp", "reference", 0]
title_name_list = ["yacht","ENB/h","ENB/c","airfoil","CCS","wine/r","wine/w","CCPP","protein","SC","CT","MSD"]
minibatch_size_list = [246,614,614,1202,824,1279,3918,7654,36584,17010,42800,463715]
NN_size_list = ["64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","64+64","128+128","256+256","64+64+64+64"]
layer_width_list = [64,64,64,64,64,64,64,64,64,128,256,64]
RUNS_list = [100,100,100,100,100,100,100,100,100,100,100,10]

label_name = label_name_list[dataset_index]
dataset_name = dataset_name_list[dataset_index]
title_name = title_name_list[dataset_index]
minibatch_size = minibatch_size_list[dataset_index]
NN_size = NN_size_list[dataset_index]
layer_width = layer_width_list[dataset_index]
RUNS = RUNS_list[dataset_index]

dataset = dataset_read(dataset_name)
print(dataset)

def build_model(layer_width):
  model = keras.Sequential([
    layers.Dense(layer_width, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(layer_width, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

def build_model_MSD(layer_width):
  model = keras.Sequential([
    layers.Dense(layer_width, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(layer_width, activation=tf.nn.relu),
    layers.Dense(layer_width, activation=tf.nn.relu),
    layers.Dense(layer_width, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

MAE_NN = []
Storage_test_labels = []
Storage_test_NN_predictions = []
MAE_GPcorrected = []
MAE_GPcorrected_inputOnly = []
MAE_GPcorrected_outputOnly = []
MAE_GP = []
MAE_GP_inputOnly = []
MAE_GP_outputOnly = []
PCT_within95Interval_GPcorrected = []
PCT_within90Interval_GPcorrected = []
PCT_within68Interval_GPcorrected = []
PCT_within95Interval_GPcorrected_inputOnly = []
PCT_within90Interval_GPcorrected_inputOnly = []
PCT_within68Interval_GPcorrected_inputOnly = []
PCT_within95Interval_GPcorrected_outputOnly = []
PCT_within90Interval_GPcorrected_outputOnly = []
PCT_within68Interval_GPcorrected_outputOnly = []
PCT_within95Interval_GP = []
PCT_within90Interval_GP = []
PCT_within68Interval_GP = []
PCT_within95Interval_GP_inputOnly = []
PCT_within90Interval_GP_inputOnly = []
PCT_within68Interval_GP_inputOnly = []
PCT_within95Interval_GP_outputOnly = []
PCT_within90Interval_GP_outputOnly = []
PCT_within68Interval_GP_outputOnly = []
Storage_mean_GPcorrected = []
Storage_var_GPcorrected = []
Storage_mean_GPcorrected_inputOnly = []
Storage_var_GPcorrected_inputOnly = []
Storage_mean_GPcorrected_outputOnly = []
Storage_var_GPcorrected_outputOnly = []
Storage_mean_GP = []
Storage_var_GP = []
Storage_mean_GP_inputOnly = []
Storage_var_GP_inputOnly = []
Storage_mean_GP_outputOnly = []
Storage_var_GP_outputOnly = []
computation_time_NN = []
computation_time_GPcorrected = []
computation_time_GPcorrected_inputOnly = []
computation_time_GPcorrected_outputOnly = []
computation_time_GP = []
computation_time_GP_inputOnly = []
computation_time_GP_outputOnly = []
hyperparameter_GPcorrected = []
hyperparameter_GPcorrected_inputOnly = []
hyperparameter_GPcorrected_outputOnly = []
hyperparameter_GP = []
hyperparameter_GP_inputOnly = []
hyperparameter_GP_outputOnly = []

for run in range(RUNS):
    tf.reset_default_graph()
    with tf.Session(graph=tf.Graph()):
        # preprocess data
        if dataset_name == "MSD":
            train_dataset = dataset.head(463715)
            test_dataset = dataset.tail(51630)
        else:
            train_dataset = dataset.sample(frac=0.8,random_state=run)
            test_dataset = dataset.drop(train_dataset.index)

        train_stats = train_dataset.describe()
        train_stats.pop(label_name)
        train_stats = train_stats.transpose()

        train_labels = train_dataset.pop(label_name)
        test_labels = test_dataset.pop(label_name)

        normed_train_data = (train_dataset - train_stats['mean']) / train_stats['std']
        normed_test_data = (test_dataset - train_stats['mean']) / train_stats['std']

        minibatch_size = len(normed_train_data)
        # training NN
        time_checkpoint1 = time.time()

        if dataset_name == "MSD":
            model = build_model_MSD(layer_width)
        else:
            model = build_model(layer_width)

        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                            validation_split = 0.2, verbose=0, callbacks=[early_stop])
        time_checkpoint2 = time.time()

        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
        computation_time_NN.append(time_checkpoint2-time_checkpoint1)
        print("computation_time_NN: {}".format(time_checkpoint2-time_checkpoint1))
        print("Testing set Mean Abs Error: {} {}".format(mae, label_name))

        MAE_NN.append(mae)

        test_NN_predictions = model.predict(normed_test_data).flatten()
        test_NN_errors = test_labels - test_NN_predictions

        train_NN_predictions = model.predict(normed_train_data).flatten()
        Storage_test_labels.append(test_labels.values)
        Storage_test_NN_predictions.append(test_NN_predictions)

        # running RIO
        kernel_type = "RBF+RBF"
        framework_variant = "GP_corrected"
        MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter = RIO_variants_running(framework_variant, \
                                                                                                                                                  kernel_type, \
                                                                                                                                                  normed_train_data, \
                                                                                                                                                  normed_test_data, \
                                                                                                                                                  train_labels, \
                                                                                                                                                  test_labels, \
                                                                                                                                                  train_NN_predictions, \
                                                                                                                                                  test_NN_predictions, \
                                                                                                                                                  M)
        MAE_GPcorrected.append(MAE)
        PCT_within95Interval_GPcorrected.append(PCT_within95Interval)
        PCT_within90Interval_GPcorrected.append(PCT_within90Interval)
        PCT_within68Interval_GPcorrected.append(PCT_within68Interval)
        Storage_mean_GPcorrected.append(mean)
        Storage_var_GPcorrected.append(var)
        computation_time_GPcorrected.append(computation_time)
        hyperparameter_GPcorrected.append(hyperparameter)

        # running "R+I" variant
        kernel_type = "RBF"
        framework_variant = "GP_corrected_inputOnly"
        MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter = RIO_variants_running(framework_variant, \
                                                                                                                                                  kernel_type, \
                                                                                                                                                  normed_train_data, \
                                                                                                                                                  normed_test_data, \
                                                                                                                                                  train_labels, \
                                                                                                                                                  test_labels, \
                                                                                                                                                  train_NN_predictions, \
                                                                                                                                                  test_NN_predictions, \
                                                                                                                                                  M)
        MAE_GPcorrected_inputOnly.append(MAE)
        PCT_within95Interval_GPcorrected_inputOnly.append(PCT_within95Interval)
        PCT_within90Interval_GPcorrected_inputOnly.append(PCT_within90Interval)
        PCT_within68Interval_GPcorrected_inputOnly.append(PCT_within68Interval)
        Storage_mean_GPcorrected_inputOnly.append(mean)
        Storage_var_GPcorrected_inputOnly.append(var)
        computation_time_GPcorrected_inputOnly.append(computation_time)
        hyperparameter_GPcorrected_inputOnly.append(hyperparameter)

        # running "R+O" variant
        kernel_type = "RBFY"
        framework_variant = "GP_corrected_outputOnly"
        MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter = RIO_variants_running(framework_variant, \
                                                                                                                                                  kernel_type, \
                                                                                                                                                  normed_train_data, \
                                                                                                                                                  normed_test_data, \
                                                                                                                                                  train_labels, \
                                                                                                                                                  test_labels, \
                                                                                                                                                  train_NN_predictions, \
                                                                                                                                                  test_NN_predictions, \
                                                                                                                                                  M)
        MAE_GPcorrected_outputOnly.append(MAE)
        PCT_within95Interval_GPcorrected_outputOnly.append(PCT_within95Interval)
        PCT_within90Interval_GPcorrected_outputOnly.append(PCT_within90Interval)
        PCT_within68Interval_GPcorrected_outputOnly.append(PCT_within68Interval)
        Storage_mean_GPcorrected_outputOnly.append(mean)
        Storage_var_GPcorrected_outputOnly.append(var)
        computation_time_GPcorrected_outputOnly.append(computation_time)
        hyperparameter_GPcorrected_outputOnly.append(hyperparameter)

        # running "Y+IO" variant
        kernel_type = "RBF+RBF"
        framework_variant = "GP"
        MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter = RIO_variants_running(framework_variant, \
                                                                                                                                                  kernel_type, \
                                                                                                                                                  normed_train_data, \
                                                                                                                                                  normed_test_data, \
                                                                                                                                                  train_labels, \
                                                                                                                                                  test_labels, \
                                                                                                                                                  train_NN_predictions, \
                                                                                                                                                  test_NN_predictions, \
                                                                                                                                                  M)
        MAE_GP.append(MAE)
        PCT_within95Interval_GP.append(PCT_within95Interval)
        PCT_within90Interval_GP.append(PCT_within90Interval)
        PCT_within68Interval_GP.append(PCT_within68Interval)
        Storage_mean_GP.append(mean)
        Storage_var_GP.append(var)
        computation_time_GP.append(computation_time)
        hyperparameter_GP.append(hyperparameter)

        # running "Y+I" variant, i.e., original SVGP
        kernel_type = "RBF"
        framework_variant = "GP_inputOnly"
        MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter = RIO_variants_running(framework_variant, \
                                                                                                                                                  kernel_type, \
                                                                                                                                                  normed_train_data, \
                                                                                                                                                  normed_test_data, \
                                                                                                                                                  train_labels, \
                                                                                                                                                  test_labels, \
                                                                                                                                                  train_NN_predictions, \
                                                                                                                                                  test_NN_predictions, \
                                                                                                                                                  M)
        MAE_GP_inputOnly.append(MAE)
        PCT_within95Interval_GP_inputOnly.append(PCT_within95Interval)
        PCT_within90Interval_GP_inputOnly.append(PCT_within90Interval)
        PCT_within68Interval_GP_inputOnly.append(PCT_within68Interval)
        Storage_mean_GP_inputOnly.append(mean)
        Storage_var_GP_inputOnly.append(var)
        computation_time_GP_inputOnly.append(computation_time)
        hyperparameter_GP_inputOnly.append(hyperparameter)

        # running "Y+O" variant
        kernel_type = "RBFY"
        framework_variant = "GP_outputOnly"
        MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter = RIO_variants_running(framework_variant, \
                                                                                                                                                  kernel_type, \
                                                                                                                                                  normed_train_data, \
                                                                                                                                                  normed_test_data, \
                                                                                                                                                  train_labels, \
                                                                                                                                                  test_labels, \
                                                                                                                                                  train_NN_predictions, \
                                                                                                                                                  test_NN_predictions, \
                                                                                                                                                  M)
        MAE_GP_outputOnly.append(MAE)
        PCT_within95Interval_GP_outputOnly.append(PCT_within95Interval)
        PCT_within90Interval_GP_outputOnly.append(PCT_within90Interval)
        PCT_within68Interval_GP_outputOnly.append(PCT_within68Interval)
        Storage_mean_GP_outputOnly.append(mean)
        Storage_var_GP_outputOnly.append(var)
        computation_time_GP_outputOnly.append(computation_time)
        hyperparameter_GP_outputOnly.append(hyperparameter)


# Saving experimental results
kernel_type = "RBF+RBF"
optimizer_name = "LBFGSB"
framework_variant = "GP_corrected"

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_NN_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(MAE_NN, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_test_labels_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_test_labels, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_test_NN_predictions_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_test_NN_predictions, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_NN_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(computation_time_NN, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(MAE_GPcorrected, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within95Interval_GPcorrected, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within90Interval_GPcorrected, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within68Interval_GPcorrected, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_mean_GPcorrected, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_var_GPcorrected, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(computation_time_GPcorrected, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(hyperparameter_GPcorrected, result_file)

kernel_type = "RBF"
framework_variant = "GP_corrected_inputOnly"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(MAE_GPcorrected_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within95Interval_GPcorrected_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within90Interval_GPcorrected_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within68Interval_GPcorrected_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_mean_GPcorrected_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_var_GPcorrected_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(computation_time_GPcorrected_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(hyperparameter_GPcorrected_inputOnly, result_file)

kernel_type = "RBFY"
framework_variant = "GP_corrected_outputOnly"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(MAE_GPcorrected_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within95Interval_GPcorrected_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within90Interval_GPcorrected_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within68Interval_GPcorrected_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_mean_GPcorrected_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_var_GPcorrected_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(computation_time_GPcorrected_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(hyperparameter_GPcorrected_outputOnly, result_file)

kernel_type = "RBF+RBF"
framework_variant = "GP"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(MAE_GP, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within95Interval_GP, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within90Interval_GP, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within68Interval_GP, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_mean_GP, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_var_GP, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(computation_time_GP, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(hyperparameter_GP, result_file)

kernel_type = "RBF"
framework_variant = "GP_inputOnly"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(MAE_GP_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within95Interval_GP_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within90Interval_GP_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within68Interval_GP_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_mean_GP_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_var_GP_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(computation_time_GP_inputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(hyperparameter_GP_inputOnly, result_file)

kernel_type = "RBFY"
framework_variant = "GP_outputOnly"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(MAE_GP_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within95Interval_GP_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within90Interval_GP_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(PCT_within68Interval_GP_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_mean_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_mean_GP_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Storage_var_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(Storage_var_GP_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Computation_time_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(computation_time_GP_outputOnly, result_file)

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','Kernel_hyperparameter_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pkl'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(hyperparameter_GP_outputOnly, result_file)


kernel_type = "RBF+RBF"
framework_variant = "GP_corrected"
f = plt.figure()
plt.scatter(MAE_NN, MAE_GPcorrected)
plt.xlabel('NN MAE [{}]'.format(label_name))
plt.ylabel('GP Corrected MAE [{}]'.format(label_name))
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100])
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBF"
framework_variant = "GP_corrected_inputOnly"
f = plt.figure()
plt.scatter(MAE_NN, MAE_GPcorrected_inputOnly)
plt.xlabel('NN MAE [{}]'.format(label_name))
plt.ylabel('GP Corrected_inputOnly MAE [{}]'.format(label_name))
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100])
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBFY"
framework_variant = "GP_corrected_outputOnly"
f = plt.figure()
plt.scatter(MAE_NN, MAE_GPcorrected_outputOnly)
plt.xlabel('NN MAE [{}]'.format(label_name))
plt.ylabel('GP Corrected_outputOnly MAE [{}]'.format(label_name))
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100])
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBF+RBF"
framework_variant = "GP"
f = plt.figure()
plt.scatter(MAE_NN, MAE_GP)
plt.xlabel('NN MAE [{}]'.format(label_name))
plt.ylabel('GP MAE [{}]'.format(label_name))
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100])
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBF"
framework_variant = "GP_inputOnly"
f = plt.figure()
plt.scatter(MAE_NN, MAE_GP_inputOnly)
plt.xlabel('NN MAE [{}]'.format(label_name))
plt.ylabel('GP_inputOnly MAE [{}]'.format(label_name))
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100])
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBFY"
framework_variant = "GP_outputOnly"
f = plt.figure()
plt.scatter(MAE_NN, MAE_GP_outputOnly)
plt.xlabel('NN MAE [{}]'.format(label_name))
plt.ylabel('GP_outputOnly MAE [{}]'.format(label_name))
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100])
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','MAE_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBF+RBF"
framework_variant = "GP_corrected"
f = plt.figure()
plt.boxplot(PCT_within95Interval_GPcorrected)
plt.ylabel('percentage of test points within 95 confidence interval (GPcorrected)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within90Interval_GPcorrected)
plt.ylabel('percentage of test points within 90 confidence interval (GPcorrected)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within68Interval_GPcorrected)
plt.ylabel('percentage of test points within 68 confidence interval (GPcorrected)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBF"
framework_variant = "GP_corrected_inputOnly"
f = plt.figure()
plt.boxplot(PCT_within95Interval_GPcorrected_inputOnly)
plt.ylabel('percentage of test points within 95 confidence interval (GPcorrected_inputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within90Interval_GPcorrected_inputOnly)
plt.ylabel('percentage of test points within 90 confidence interval (GPcorrected_inputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within68Interval_GPcorrected_inputOnly)
plt.ylabel('percentage of test points within 68 confidence interval (GPcorrected_inputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBFY"
framework_variant = "GP_corrected_outputOnly"
f = plt.figure()
plt.boxplot(PCT_within95Interval_GPcorrected_outputOnly)
plt.ylabel('percentage of test points within 95 confidence interval (GPcorrected_outputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within90Interval_GPcorrected_outputOnly)
plt.ylabel('percentage of test points within 90 confidence interval (GPcorrected_outputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within68Interval_GPcorrected_outputOnly)
plt.ylabel('percentage of test points within 68 confidence interval (GPcorrected_outputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBF+RBF"
framework_variant = "GP"
f = plt.figure()
plt.boxplot(PCT_within95Interval_GP)
plt.ylabel('percentage of test points within 95 confidence interval (GP)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within90Interval_GP)
plt.ylabel('percentage of test points within 90 confidence interval (GP)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within68Interval_GP)
plt.ylabel('percentage of test points within 68 confidence interval (GP)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBF"
framework_variant = "GP_inputOnly"
f = plt.figure()
plt.boxplot(PCT_within95Interval_GP_inputOnly)
plt.ylabel('percentage of test points within 95 confidence interval (GP_inputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within90Interval_GP_inputOnly)
plt.ylabel('percentage of test points within 90 confidence interval (GP_inputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within68Interval_GP_inputOnly)
plt.ylabel('percentage of test points within 68 confidence interval (GP_inputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

kernel_type = "RBFY"
framework_variant = "GP_outputOnly"
f = plt.figure()
plt.boxplot(PCT_within95Interval_GP_outputOnly)
plt.ylabel('percentage of test points within 95 confidence interval (GP_outputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within95Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within90Interval_GP_outputOnly)
plt.ylabel('percentage of test points within 90 confidence interval (GP_outputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within90Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plt.boxplot(PCT_within68Interval_GP_outputOnly)
plt.ylabel('percentage of test points within 68 confidence interval (GP_outputOnly)')
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','PCT_within68Interval_{}_suffledData_{}_{}_{}_{}_M{}_minibatch{}_{}_{}run.pdf'.format(framework_variant, dataset_name, model_name, NN_size, kernel_type, M, minibatch_size, optimizer_name, RUNS))
f.savefig(plot_file_name, bbox_inches='tight')

plt.show()
