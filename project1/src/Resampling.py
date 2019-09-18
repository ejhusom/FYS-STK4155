#!/usr/bin/env python3
# ============================================================================
# File:     Resampling.py
# Author:   Erik Johannes Husom
# Created:  2019-09-11
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
from Regression import *
from sklearn.model_selection import KFold

class Resampling():

    def __init__(self, method='ols', lambda_=0):

        self.method = method
        self.lambda_ = lambda_
        self.mse_train = None 
        self.mse_test = None 
        self.r2_train = None
        self.r2_test = None


    def cross_validation(self, X, y, n_folds):

        if len(y.shape) > 1:
            y = np.ravel(y)

        kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)

        mse = np.zeros((n_folds, 2))
        r2 = np.zeros((n_folds, 2))

        

        i = 0
        for train_index, test_index in kf.split(X):
            model = Regression(self.method, self.lambda_)
            model.fit(X[train_index], y[train_index])

            model.predict(X[train_index])
            y_predict_train = model.y_predict
            
            model.predict(X[test_index])
            y_predict_test = model.y_predict


            mse[i][0] = mean_squared_error(y[train_index], y_predict_train)
            mse[i][1] = mean_squared_error(y[test_index], y_predict_test)
            r2[i][0] = r2_score(y[train_index], y_predict_train)
            r2[i][1] = r2_score(y[test_index], y_predict_test)

            i += 1

        self.mse_train = np.mean(mse[:,0])
        self.mse_test = np.mean(mse[:,1])
        self.r2_train = np.mean(r2[:,0])
        self.r2_test = np.mean(r2[:,1])
