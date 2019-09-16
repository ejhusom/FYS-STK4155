#!/usr/bin/env python3
# ============================================================================
# File:     test_Regression.py
# Author:   Erik Johannes Husom
# Created:  2019-09-16
# ----------------------------------------------------------------------------
# Description:
# Testing the Regression.py class.
# ============================================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the source code directory to python path in order to import the code
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from Regression import *


def test_Regression_fit(method='ols'):
    
    # Data generation
    N = 6 # data size
    p = 2   # polynomial degree

    np.random.seed(0)
    x = np.random.rand(N, 1)
    y = 5*x*x + 0.1*np.random.randn(N, 1)


    # Creating design matrix X
    X = np.ones((N, p + 1))
    for i in range(1, p + 1):
        X[:,i] = x[:,0]**i


    test_model = Regression(method=method, lambda_=0.1)

    # Manual
    test_model.fit(X, y)
    beta = test_model.beta
    test_model.predict(X)
    y_tilde = test_model.z_tilde

    print(test_model.get_r2())

    # Scikit-learn
    test_model.skl_fit(X, y)
    beta_skl = test_model.beta
    test_model.skl_predict(X)
    y_tilde_skl = test_model.z_tilde
    print(test_model.get_r2())

    print(beta)
    print(beta_skl)
    print(y_tilde)
    print(y_tilde_skl)

    tol = 1e-10

    assert np.max(abs(beta - beta_skl)) < tol
    assert np.max(abs(y_tilde - y_tilde_skl)) < tol

    plot_regression(x, y, x, y_tilde)



def plot_regression(x, y, x_fit, y_fit, y_label='model fit'):
    '''Plot data and model fit in one figure.'''
    plt.figure()
    plt.plot(x, y, '.', label='random data')
    plt.plot(x_fit, y_fit, '-', label=y_label)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    test_Regression_fit('ols')
    test_Regression_fit('ridge')
    test_Regression_fit('lasso')
