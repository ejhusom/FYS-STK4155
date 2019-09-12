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

class Resampling(Regression):


    def cross_validation(self, n_folds, method='ols', lambda_=0.1):

        kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)

        mse = np.zeros((n_folds, 2))
        r2 = np.zeros((n_folds, 2))

        i = 0
        for train_index, test_index in kf.split(self.X):

            if method=='ridge':
                self.ridge(lambda_)
            elif method=='lasso':
                self.lasso(lambda_)
            else:
                self.ols()


            z_tilde_train = self.X[train_index] @ self.beta
            z_tilde_test = self.X[test_index] @ self.beta


            mse[i][0] = mean_squared_error(self.z[train_index], z_tilde_train)
            mse[i][1] = mean_squared_error(self.z[test_index], z_tilde_test)
            r2[i][0] = r2_score(self.z[train_index], z_tilde_train)
            r2[i][1] = r2_score(self.z[test_index], z_tilde_test)

            i += 1

        self.mse_train = np.mean(mse[:,0])
        self.mse_test = np.mean(mse[:,1])
        self.r2_train = np.mean(r2[:,0])
        self.r2_test = np.mean(r2[:,1])
