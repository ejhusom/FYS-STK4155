#!/usr/bin/env python3
# ============================================================================
# File:     metrics.py
# Author:   Erik Johannes Husom
# Created:  2019-10-04
# ----------------------------------------------------------------------------
# Description:
# Statistical metrics for machine learning.
# ============================================================================
import numpy as np

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def bias(y_true, y_pred):
    return np.mean((y_true - np.mean(y_pred))**2)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2, axis=0)

def r2_score(y_true, y_pred):
    
    TSS = np.sum((y_true - np.mean(y_true))**2) # total sum of squares
    RSS = np.sum((y_true - y_pred)**2)          # residual sum of squares

    return 1 - RSS/TSS
