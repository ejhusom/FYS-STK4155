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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
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
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        mse[i] = mean_squared_error(y[val_idx], y_pred)
        r2[i] = r2_score(y[val_idx], y_pred)
        accuracy[i] = accuracy_score(y[val_idx], y_pred)

        i += 1

    mse_cv = np.mean(mse)
    r2_cv = np.mean(r2)
    accuracy_cv = np.mean(accuracy)

    return mse_cv, r2_cv, accuracy_cv
