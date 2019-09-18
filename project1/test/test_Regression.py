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
from franke import *
from designmatrix import *


def test_Regression_fit(method='ols'):
    
    # Data generation
#    N = 100 # data size
#    p = 5   # polynomial degree
#
#    np.random.seed(0)
#    x = np.random.rand(N, 1)
#    y = 5*x*x + 0.1*np.random.randn(N, 1)
#
#
#    # Creating design matrix X
#    X = np.ones((N, p + 1))
#    for i in range(1, p + 1):
#        X[:,i] = x[:,0]**i
#
    x, y = franke.generate_xy(0, 1, 100)
    z = franke.franke_function(x, y, eps=0.00)

    X = designmatrix.create_design_matrix(x, y, deg=5)
    
    test_model = Regression(method=method, lambda_=0.01)

    # Manual
    test_model.fit(X, y)
    beta = test_model.beta
    test_model.predict(X)
    y_predict = test_model.z_predict
    print('R2 manual:')
    print(test_model.get_r2())

    # Scikit-learn
    test_model.skl_fit(X, y)
    beta_skl = test_model.beta
    test_model.skl_predict(X)
    y_predict_skl = test_model.z_predict
    print('R2 scikit:')
    print(test_model.get_r2())

    print('Beta:')
    print(beta)
    print(beta_skl)
    print('y:')
    print(y_predict)
    print(y_predict_skl)

    tol = 1e-8

    assert np.max(abs(beta - beta_skl)) < tol
    assert np.max(abs(y_predict - y_predict_skl)) < tol

    #plot_regression(x, y, x, y_predict)



def plot_regression(x, y, x_fit, y_fit, y_label='model fit'):
    '''Plot data and model fit in one figure.'''
    plt.figure()
    plt.plot(x, y, '.', label='random data')
    plt.plot(x_fit, y_fit, '-', label=y_label)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    #test_Regression_fit('ols')
    test_Regression_fit('ridge')
    #test_Regression_fit('lasso')
