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
    """
    Fitting regression model on dataset with scalar response/target.

    Attributes
    ----------
        method : str, default='ols'
            Which regression method to use.
        alpha : float, default=0
            Hyperparameter for shrinkage methods.
        X : array
        y : array
        y_pred : array
        beta : array


    Methods
    -------
        fit
            Fitting model using one of the three regression methods OLS, Ridge
            or Lasso.
        predict(
            Using model to predict.
        ols
            Ordinary Least Squares regression.
        ridge
            Ridge regression.
        lasso
            Lasso regression with sklearn as backend.

    """

    def __init__(self, 
            method='ols', 
            alpha=0):

        self.method = method
        self.alpha = alpha

        self.X = None
        self.y = None
        self.y_pred = None
        self.beta = None



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
                    self.alpha*np.identity(np.shape(self.X)[1])) @ X.T @
                    self.y)

    def lasso(self):

        model = skl.Lasso(alpha=self.alpha, fit_intercept=False,
                normalize=False, max_iter=10000).fit(self.X, self.y)
        self.beta = clf.coef_




