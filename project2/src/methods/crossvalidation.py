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
from sklearn.preprocessing import StandardScaler
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
        X_train, X_val= standardize_train_val(X[train_idx], X[val_idx])
#        model.fit(X[train_idx], y[train_idx])
#        y_pred = model.predict(X[val_idx])
#        mse[i] = mean_squared_error(y[val_idx], y_pred)
#        r2[i] = r2_score(y[val_idx], y_pred)
#        accuracy[i] = accuracy_score(y[val_idx], y_pred)
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_val)
        mse[i] = mean_squared_error(y[val_idx], y_pred)
        r2[i] = r2_score(y[val_idx], y_pred)
        accuracy[i] = accuracy_score(y[val_idx], y_pred)

        i += 1

    mse_cv = np.mean(mse)
    r2_cv = np.mean(r2)
    accuracy_cv = np.mean(accuracy)

    return mse_cv, r2_cv, accuracy_cv


def standardize_train_val(X_train, X_val):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

#    X_train = X_train - np.mean(X_train, axis=0)
#    X_test = X_test - np.mean(X_train, axis=0)
#    col_std = np.std(X_train, axis=0)
#
#    for i in range(0, np.shape(X_train)[1]):
#        X_train[:,i] /= col_std[i]
#        X_test[:,i] /= col_std[i]


    return X_train, X_val
