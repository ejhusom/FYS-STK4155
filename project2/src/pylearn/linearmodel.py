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


class Regression():

    def __init__(self, method='ols', lambda_=0):


        self.method = method

        self.X = None
        self.y = None
        self.lambda_ = lambda_

        self.y_pred = None
        self.beta = None

        self.skl_model = None


    def fit(self, X, y):
        
        self.X = X
        self.y = y

        if self.method == 'ols':
            self.ols()
        elif self.method == 'ridge':
            self.ridge()
        elif self.method == 'lasso':
            self.skl_fit(X, y)


    def predict(self, X=None, beta=None):
        if X is None:
            X = self.X
        if beta is None:
            beta = self.beta

        self.y_pred = X @ beta

        return self.y_pred


    def ols(self):
        '''Ordinary least squares.'''
        X = self.X
        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(self.y)


    def ridge(self):

        X = self.X

        self.beta = (np.linalg.pinv(X.T.dot(X) +
                    self.lambda_*np.identity(np.shape(self.X)[1])) @ X.T @
                    self.y)

    def lasso(self):

        model = skl.Lasso(alpha=self.lambda_, fit_intercept=False,
                normalize=False, max_iter=10000).fit(self.X, self.y)
        self.beta = clf.coef_




