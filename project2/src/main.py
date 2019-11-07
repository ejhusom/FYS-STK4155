#!/usr/bin/env python3
# ============================================================================
# File:     main.py
# Author:   Erik Johannes Husom
# Created:  2019-11-06
# ----------------------------------------------------------------------------
# Description:
# Main function for running analysis of regression and classification methods
# in FYS-STK4155 project 2, fall 2019.
# ============================================================================
from nn_regression import *
from nn_classification import *


np.random.seed(2010)



# ============================================================================
# REGRESSION

# nn_regression_analysis(train=False)
# nn_regression_heatmap(train=True)
# nn_regression_optimal(train=False)
# nn_regression_gridsize(train=True)

# ============================================================================
# CLASSIFICATION

# nn_classification_simple()
# nn_classification_skl()
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

# nn_classification_optimal(train=True, options=[2, True], layers=3, nodes=80, eta=1e-1)


# ============================================================================
# BENCHMARK SCIKIT-LEARN

