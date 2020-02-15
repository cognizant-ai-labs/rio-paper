RIO Server Software
Copyright (C) 2020 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.

# rio-paper
Code and supporting materials for the ICLR 2020 RIO paper

This repository contains all the source codes to reproduce the experimental results reported in paper "Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel", which is published in ICLR 2020.

To reproduce the testing environment of the source codes, use environment.yml to create an anaconda environment as follows:
$ conda env create -f environment.yml
$ conda activate RIO_experiments

Before running the codes, three directories need to be created under the current path:
./Datasets/ - contains the original datasets downloaded from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets.php?format=&task=reg&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table)
./Plots/ - for storing generated figures
./Results/ - for storing experimental results

Usages of each python file:
main_experiments_RIO_variants.py - main file to run tests for all the RIO variants on all the datasets
util.py - contains functions to read data and run RIO variants
Results_Table1.py - file for post-processing all the experimental results for Table 1 (in the main paper)
Results_Figure_RMSE.py - file for plotting all the figures in Figure 3
Results_Figure_CI.py - file for plotting all the figures in Figure 4 and Figure 5
Results_Spearman_correlation.py - file for calculating Spearman's rank correlation between RMSE and noise variance
