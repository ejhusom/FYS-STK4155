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


def bias(y_true, y_pred):
    return np.mean((y_true - np.mean(y_pred))**2)
