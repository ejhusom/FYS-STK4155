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

    def __init__(self, X, z):

        #if np.shape(z) > 1:
        z = np.ravel(z)

        self.X = X      # design matrix
        self.z = z      # response variable

    def ols(self):
        '''Ordinary least squares.'''

        X = self.X
        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(self.z)
        self.z_tilde = X @ self.beta


    def ridge(self, lambda_=0.1):

        X = self.X
        self.beta = np.linalg.pinv(X.T.dot(X) + \
            lambda_*np.identity(self.p)).dot(X.T).dot(self.z)
        self.z_tilde = X @ self.beta

    def lasso(self, lambda_=0.1):
        
        clf_lasso = skl.Lasso(alpha=lambda_).fit(self.X, self.z)
        self.beta = clf_lasso.get_params()
        self.z_tilde = clf_lasso.predict(self.X)
        
    
    def print_error_analysis(self):
        print(self.mse())
        print(self.r2())
        print(self.var_beta())


    def mse(self):
        return mean_squared_error(self.z, self.z_tilde) 


    def r2(self):
        return r2_score(self.z, self.z_tilde) 


    def var_beta(self):
        return np.var(self.z)*np.linalg.pinv(self.X.T @ self.X)

