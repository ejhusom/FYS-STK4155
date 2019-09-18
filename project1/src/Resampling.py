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

class Resampling(Regression):

    def __init__(self):

        self.mse_train = None 
        self.mse_test = None 
        self.r2_train = None
        self.r2_test = None


    def cross_validation(self, X, z, n_folds, lambda_=0.1):

        
        if len(np.shape(z)) > 1:
            z = np.ravel(z)

        self.set_lambda_ = lambda_

        kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)

        mse = np.zeros((n_folds, 2))
        r2 = np.zeros((n_folds, 2))

        

        i = 0
        for train_index, test_index in kf.split(X):
            
            self.fit(X[train_index], z[train_index])

            z_predict_train = self.z_predict
            z_predict_test = X[test_index] @ self.beta


            mse[i][0] = mean_squared_error(z[train_index], z_predict_train)
            mse[i][1] = mean_squared_error(z[test_index], z_predict_test)
            r2[i][0] = r2_score(z[train_index], z_predict_train)
            r2[i][1] = r2_score(z[test_index], z_predict_test)

            i += 1

        self.mse_train = np.mean(mse[:,0])
        self.mse_test = np.mean(mse[:,1])
        self.r2_train = np.mean(r2[:,0])
        self.r2_test = np.mean(r2[:,1])
