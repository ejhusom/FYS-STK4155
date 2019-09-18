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

def create_design_matrix(x, y, deg=5):

    p = int((deg+1)*(deg+2)/2)
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    X = np.ones((N,p))

    for i in range(1, deg+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X

