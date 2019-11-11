#!/usr/bin/env python3
# ============================================================================
# File:     main.py
# Created:  2019-11-06
# ----------------------------------------------------------------------------
# Description:
# Main function for running analysis of regression and classification methods
# in FYS-STK4155 project 2, fall 2019.
#
# USAGE:
# Each function call has an explanational comment. Uncomment function calls to
# perform the specified analysis.
# ============================================================================
from nn_regression import *
from nn_classification import *
from logistic_classification import *

np.random.seed(2010)

# ============================================================================
# REGRESSION

# Search for optimal learning rate and epoch number:
# nn_regression_analysis(train=False)

# Grid search for hidden layer configuration:
# nn_regression_heatmap(train=True)

# Train model with optimal parameters:
# nn_regression_optimal(train=True)

# Compare Scikit-Learn's implementation with pylearn:
# nn_regression_skl()

# Analyze behaviour of MSE and R2 for different gridsize of Franke data set:
# nn_regression_gridsize(train=True)

# ============================================================================
# CLASSIFICATION NEURAL NETWORK

# Simple test case for neural network classification:
# nn_classification_simple()

# Comparison of Scikit-Learn and pylearn when using breast cancer data set:
# nn_classification_skl()

# ----------------------------------------------------------------------------
# Analysis of optimal parameters when training model on credit card data set,
# by finding optimal learning rate and best configuration of hidden layers. The
# last step, in the function nn_classification_optimal() trains the neural
# network using the best parameters.

# cc_options = []
# cc_options.append([1, False])
# cc_options.append([2, False])
# cc_options.append([1, True])
# cc_options.append([2, True])

# eta_opts = []
# accuracies = []

# for o in cc_options:
#     eta_opt, accuracy = nn_classification_analysis(train=True, options=o)
#     eta_opts.append(eta_opt)
#     accuracies.append(accuracy)

# nn_classification_plot_analysis(cc_options)

# accuracies = np.array(accuracies)
# best_idx = np.argmax(accuracies)
# layers, nodes = nn_classification_heatmap(train=True, options=cc_options[best_idx],
#         eta=eta_opts[best_idx])

# nn_classification_optimal(train=True, options=cc_options[best_idx],
#         eta=eta_opts[best_idx], layers=layers, nodes=nodes)


# ============================================================================
# CLASSIFICATION LOGISTIC REGRESSION

# filename = 'data/credit_card.xls'
# X, y, scale_columns = preprocess_CC_data(filename, which_onehot = 2)
# model = SGDClassification(batch_size=100, eta0=0.001)
# analyze_logistic(X, y, model, scale_columns, analyze_params=True, balance_outcomes=False)
