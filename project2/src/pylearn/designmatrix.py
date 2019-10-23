#!/usr/bin/env python3
# ============================================================================
# File:     designmatrix.py
# Author:   Erik Johannes Husom
# Created:  2019-09-18
# ----------------------------------------------------------------------------
# Description:
# Create design matrix.
# ============================================================================
import numpy as np

def create_design_matrix(x1, x2, deg=5):

    p = int((deg+1)*(deg+2)/2)
    if len(x1.shape) > 1:
        x1 = np.ravel(x1)
        x2 = np.ravel(x2)

    N = len(x1)
    X = np.ones((N,p))

    for i in range(1, deg+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x1**(i-k) * x2**k

    return X

