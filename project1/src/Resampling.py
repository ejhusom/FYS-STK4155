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

    def cross_validation(self):

        kf = KFold(n_splits=5, random_state=0, shuffle=True)

        for train_index, test_index in kf.split(self.X):

            beta, z_OLS = self.regression(self.X[train_index], self.z[train_index],
                    lmd=0)

            z_tilde = self.X[test_index] @ beta

            self.print_error_analysis(self.z[test_index], z_tilde)
