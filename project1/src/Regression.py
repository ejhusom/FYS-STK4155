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

    def __init__(self, method='ols', lambda_=0):


        self.method = method

        self.X = None
        self.y = None
        self.lambda_ = lambda_


        self.y_predict = None
        self.beta = None
        self.mse = None
        self.r2 = None
        self.beta_var = None

        self.skl_model = None


    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def fit(self, X, y):
        
        if len(np.shape(y)) > 1:
            y = np.ravel(y)

        self.X = X
        self.y = y

        if self.method == 'ols':
            self.ols()
        elif self.method == 'ridge':
            self.ridge()
        elif self.method == 'lasso':
            self.skl_fit(X, y)


    def predict(self, X):

        
        if self.method == 'ridge':
            #X -= np.mean(self.X, axis=0)
            print(np.mean(self.X, axis=0))
            self.y_predict = X @ self.beta + np.mean(self.y)
        else:
            self.y_predict = X @ self.beta

    def ols(self):
        '''Ordinary least squares.'''
        #X += 0.1
        X = self.X
        #XTX = X.T.dot(X)
        #Xinv = np.linalg.pinv(XTX)
        #XinvXT = Xinv @ X.T
        #self.beta = XinvXT @ self.y
        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(self.y)


    def ridge(self):

        # Centering data
        self.X -= np.mean(self.X, axis=0)
        self.y -= np.mean(self.y)

# TODO: Normalize
#        col_var = np.var(self.X, axis=0)
#
#        for i in range(1, len(self.X[0,:])):
#            self.X[:,i] /= col_var[i]
#
#        self.X /= np.var(self.X, axis=0)
#        self.y /= np.var(self.y)


        X = self.X
        self.beta = np.linalg.pinv(X.T.dot(X) + \
            self.lambda_*np.identity(np.shape(self.X)[1])).dot(X.T) @ self.y

    def skl_fit(self, X, y):


        self.X = X
        
        if len(y.shape) > 1:
            y = np.ravel(y)
        
        self.y = y

        if self.method == 'ols':
            self.skl_model = skl.LinearRegression()
        elif self.method == 'ridge':
            self.skl_model = skl.Ridge(alpha=self.lambda_, fit_intercept=True)
        elif self.method == 'lasso':
            self.skl_model = skl.Lasso(alpha=self.lambda_, fit_intercept=True)

        self.skl_model.fit(self.X, self.y)
        if len(np.shape(self.skl_model.coef_)) > 1:
            self.beta = self.skl_model.coef_[0]
        else:
            self.beta = self.skl_model.coef_
        self.beta[0] = self.skl_model.intercept_


    def skl_predict(self, X):

        self.y_predict = np.ravel(self.skl_model.predict(X)) #- self.beta[0])


    
    def print_error_analysis(self):
        print(self.get_mse())
        print(self.get_r2())
        #print(self.get_var_beta())


    def get_mse(self):
        self.mse = np.sum((self.y - self.y_predict)**2) / self.y.size

        return self.mse

    def get_r2(self):
        return r2_score(self.y, self.y_predict) 


    def get_var_beta(self):
        return np.var(self.y)*np.linalg.pinv(self.X.T @ self.X)

