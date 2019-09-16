#!/usr/bin/env python3
# ============================================================================
# File:     Regression.py
# Author:   Erik Johannes Husom
# Created:  2019-09-11
# ----------------------------------------------------------------------------
# Description:
# Class for linear regression and resampling methods
# ============================================================================
import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold



class Regression():

    def __init__(self, X, z):

        if len(np.shape(z)) > 1:
            z = np.ravel(z)

        self.X = X      # design matrix
        self.z = z      # response variable

    def ols(self):
        '''Ordinary least squares.'''

        X = self.X
        XTX = X.T.dot(X)
        Xinv = np.linalg.pinv(XTX)
        XinvXT = Xinv @ X.T
        self.beta = XinvXT @ self.z
        #self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(self.z)
        self.z_tilde = X @ self.beta


    def ridge(self, lambda_=0.1):

        self.X -= np.mean(self.X, axis=0)
        self.z -= np.mean(self.z)


        X = self.X
        self.beta = np.linalg.pinv(X.T.dot(X) + \
            lambda_*np.identity(np.shape(self.X)[1])).dot(X.T).dot(self.z)
        self.z_tilde = X @ self.beta + np.mean(self.z)
        print(self.beta)



    def skl_ridge(self):

        lambda_ = 0.1
        clf_ridge = skl.Ridge(alpha=lambda_)
        clf_ridge.fit(self.X, self.z)
        self.beta = clf_ridge.coef_
        print(self.beta)
        self.z_tilde = clf_ridge.predict(self.X)


    def lasso(self, lambda_=0.1):

        clf_lasso = skl.Lasso(alpha=lambda_, fit_intercept=False)
        clf_lasso.fit(self.X, self.z)
        self.beta = clf_lasso.coef_
        print(self.beta)
        self.z_tilde = clf_lasso.predict(self.X)
        
    
    def print_error_analysis(self):
        print(self.get_mse())
        print(self.get_r2())
        #print(self.get_var_beta())


    def get_mse(self):
        return mean_squared_error(self.z, self.z_tilde) 


    def get_r2(self):
        return r2_score(self.z, self.z_tilde) 


    def get_var_beta(self):
        return np.var(self.z)*np.linalg.pinv(self.X.T @ self.X)

