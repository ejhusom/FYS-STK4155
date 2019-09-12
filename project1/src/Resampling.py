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

        for train_index, test_index in kf.split(self.X):

            if method==0:
                self.ols()
            elif method==1:
                self.ridge()
            else:
                self.lasso()

            z_tilde = self.X[test_index] @ self.beta

            self.print_error_analysis()
