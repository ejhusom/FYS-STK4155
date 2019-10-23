#!/usr/bin/env python3
# ============================================================================
# File:     resampling.py
# Author:   Erik Johannes Husom
# Created:  2019-10-23
# Version:  2.0
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from pylearn.logisticregression import SGDClassification
from pylearn.linearmodel import Regression
from pylearn.metrics import *
from pylearn.neuralnetwork import NeuralNetwork


def CV(X, y, model, n_splits=5, random_state=0, classification=True):

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    mse = np.zeros(n_splits)
    r2 = np.zeros(n_splits)
    accuracy = np.zeros(n_splits)

    i = 0
    for train_idx, val_idx in kf.split(X):
        X_train, X_val= standardize_train_val(X[train_idx], X[val_idx])

        # Scaling target y if not classification problem
        if not classification:
            y_train, y_val = standardize_train_val(y[train_idx], y[val_idx])
        else:
            y_train, y_val = y[train_idx], y[val_idx]

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Store performance cores
        mse[i] = mean_squared_error(y_val, y_pred)
        r2[i] = r2_score(y_val, y_pred)
        # Only include accuracy if the model is a classification problem
        if classification:
            accuracy[i] = accuracy_score(y_val, y_pred)

        i += 1

    # Calculate mean of performance scores
    mse_cv = np.mean(mse)
    r2_cv = np.mean(r2)
    accuracy_cv = np.mean(accuracy)

    return mse_cv, r2_cv, accuracy_cv


def standardize_train_val(train, val):

    sc = StandardScaler()
    train = sc.fit_transform(train)
    val = sc.transform(val)

# TODO: Manual scaling
#    X_train = X_train - np.mean(X_train, axis=0)
#    X_test = X_test - np.mean(X_train, axis=0)
#    col_std = np.std(X_train, axis=0)
#
#    for i in range(0, np.shape(X_train)[1]):
#        X_train[:,i] /= col_std[i]
#        X_test[:,i] /= col_std[i]


    return train, val
