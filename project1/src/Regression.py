#!/usr/bin/env python3
# ============================================================================
# File:     RegressionResampling.py
# Author:   Erik Johannes Husom
# Created:  2019-09-11
# ----------------------------------------------------------------------------
# Description:
# Class for linear regression and resampling methods
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression



class Regression():

    def __init__(self, x, y, z, deg=5):

        self.n = len(x)
        self.deg = deg
        self.p = int((self.deg+1)*(self.deg+2)/2)
        self.x = x
        self.y = y
        self.z = z

        self.X = self.create_design_matrix()


    def franke_function(self, eps = 0.05):

        np.random.seed(0)

        x = np.reshape(self.x, (self.n, self.n))
        y = np.reshape(self.y, (self.n, self.n))

        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        
        z = term1 + term2 + term3 + term4 + eps*np.random.randn(self.n)

        return np.ravel(z)


    def generate_xy(self, start = 0, stop = 1):
        '''Generate x and y data and return at as a flat meshgrid.'''

        x = np.linspace(start, stop, self.n)
        y = np.linspace(start, stop, self.n)
        x, y = np.meshgrid(x, y)
        
        return np.ravel(x), np.ravel(y)


    def create_design_matrix(self):

        N = len(self.x)
        X = np.ones((N,self.p))

        for i in range(1, self.deg+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = self.x**(i-k) * self.y**k

        return X


    def regression(self, lambda_=0):
        '''
        Perform linear regression.

        lmd indicates type of regression:
        lmd=0: Ordinary least squares.
        lmd>0: Ridge regression.
        '''

        X = self.X

        self.beta = np.linalg.pinv(X.T.dot(X) + \
            lambda_*np.identity(self.p)).dot(X.T).dot(self.z)
        self.z_tilde = X @ self.beta
        self.mse = mean_squared_error(self.z, self.z_tilde)
        self.r2 = r2_score(self.z, self.z_tilde)


    def lasso(self, lambda_=0.1):
        
        clf_lasso = skl.Lasso(alpha=lambda_).fit(self.X, self.z)
        self.beta = clf_lasso.get_params()
        self.z_tilde = clf_lasso.predict(self.X)
        self.mse = mean_squared_error(self.z, self.z_tilde)
        self.r2 = r2_score(self.z, self.z_tilde)
        

    def print_error_analysis(self, z, z_tilde):
        '''Print error analysis of regression fit using scikit.'''
        
        print("Mean squared error: %.8f" % mean_squared_error(z, z_tilde))
        print('R2 score: %.8f' % r2_score(z, z_tilde))
        print('Mean absolute error: %.8f' % mean_absolute_error(z, z_tilde))

