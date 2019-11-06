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

# nn_regression_analysis(train=False)
# nn_regression_heatmap(train=True)
# nn_regression_optimal(train=False)

# nn_classification_simple()
# nn_classification_skl()
# nn_classification_analysis(train=True)
nn_classification_optimal(train=True)
