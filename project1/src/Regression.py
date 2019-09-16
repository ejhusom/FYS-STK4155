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



class Regression():

    def __init__(self, method='ols'):


        self.method = method

        self.X = None
        self.z = None
        self.lambda_ = 0


        self.z_tilde = None
        self.beta = None
        self.mse = None
        self.r2 = None
        self.beta_var = None

        self.skl_model = None


    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def fit(self, X, z):
        
        if len(np.shape(z)) > 1:
            z = np.ravel(z)

        self.X = X
        self.z = z

        if self.method == 'ols':
            self.ols()
        elif self.method == 'ridge':
            self.ridge()
        elif self.method == 'lasso':
            self.lasso()


    def predict(self, X):

        self.z_tilde = X @ self.beta
        
        if self.method == 'ridge':
            self.z_tilde += np.mean(self.z)


    def ols(self):
        '''Ordinary least squares.'''
        #X += 0.1
        X = self.X
        #XTX = X.T.dot(X)
        #Xinv = np.linalg.pinv(XTX)
        #XinvXT = Xinv @ X.T
        #self.beta = XinvXT @ self.z
        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(self.z)


    def ridge(self):

        self.X -= np.mean(self.X, axis=0)
        self.z -= np.mean(self.z)


        X = self.X
        self.beta = np.linalg.pinv(X.T.dot(X) + \
            self.lambda_*np.identity(np.shape(self.X)[1])).dot(X.T) @ self.z

    def skl_fit(self, X, z):


        self.X = X
        self.z = z

        if self.method == 'ols':
            self.skl_model = skl.LinearRegression()
        elif self.method == 'ridge':
            self.skl_model = skl.Ridge(alpha=self.lambda_)
        elif self.method == 'lasso':
            self.skl_model = skl.Lasso(alpha=self.lambda_)

        self.skl_model.fit(self.X, self.z)
        self.beta = self.skl_model.coef_[0]
        self.beta[0] = self.skl_model.intercept_


    def skl_predict(self, X):

        self.z_tilde = np.ravel(self.skl_model.predict(X) - self.beta[0])


    def lasso(self):

#       self.X = np.delete(self.X, 0, 1)
#        self.X -= np.mean(self.X, axis=0)
#        self.z -= np.mean(self.z)


        clf_lasso = skl.Lasso(alpha=self.lambda_, max_iter=100000)
        clf_lasso.fit(self.X, self.z)
        self.beta = clf_lasso.coef_
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

