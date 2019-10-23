#!/usr/bin/env python3
# ============================================================================
# File:     crossvalidation.py
# Author:   Erik Johannes Husom
# Created:  2019-10-23
# Version:  2.0
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from logisticregression import SGDClassification
from neuralnetwork import NeuralNetwork
from linearmodel import Regression


def CV(X, y, model, n_splits=5):

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    mse = np.zeros(n_splits)
    r2 = np.zeros(n_splits)
    accuracy = np.zeros(n_splits)

    i = 0
    for train_idx, val_idx in kf.split(X):
        #model = SGDClassification(eta0=0.01)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        accuracy[i] = accuracy_score(y[val_idx], y_pred)
        print(accuracy[i])
#        mse[i] = mean_squared_error(self.y[val_idx], self.y_pred)
#        r2[i] = r2_score(self.y[val_idx], self.y_pred)

        i += 1

#    self.mse_cv = np.mean(mse)
#    self.r2_cv = np.mean(r2)
    accuracy_cv = np.mean(accuracy)

    return accuracy_cv
#    return self.mse_cv, self.r2_cv
