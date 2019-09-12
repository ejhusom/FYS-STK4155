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


    def cross_validation(self, n_folds, method=0):

        kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)

        mse = np.zeros(n_folds)
        r2 = np.zeros(n_folds)

        i = 0
        for train_index, test_index in kf.split(self.X):

            if method==0:
                self.ols()
            elif method==1:
                self.ridge()
            else:
                self.lasso()

            z_tilde = self.X[test_index] @ self.beta

            mse[i] = mean_squared_error(self.z, z_tilde)
            r2[i] = r2_score(self.z, z_tilde)

            i += 1

        self.mse_cv = np.mean(mse)
        self.r2.cv = np.mean(r2)
