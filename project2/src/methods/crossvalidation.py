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
from metrics.regression import *
from sklearn.model_selection import KFold, train_test_split

def cross_validation(X, y, n_folds, method='ols', lambda_=0.01):

    if len(y.shape) > 1:
        y = np.ravel(y)

    kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)

    mse = np.zeros((n_folds, 2))
    r2 = np.zeros((n_folds, 2))
    b = np.zeros((n_folds, 2))
    var = np.zeros((n_folds, 2))

    
    i = 0
    for train_index, val_index in kf.split(X):
        model = Regression(method, lambda_)
        model.fit(X[train_index], y[train_index])

        model.predict(X[train_index])
        y_pred_train = model.y_pred
        
        model.predict(X[val_index])
        y_pred_test = model.y_pred


        mse[i][0] = mean_squared_error(y[train_index], y_pred_train)
        mse[i][1] = mean_squared_error(y[val_index], y_pred_test)
        r2[i][0] = r2_score(y[train_index], y_pred_train)
        r2[i][1] = r2_score(y[val_index], y_pred_test)
        b[i][0] = bias(y[train_index], y_pred_train)
        b[i][1] = bias(y[val_index], y_pred_test)
        var[i][0] = np.var(y_pred_train)
        var[i][1] = np.var(y_pred_test)

        i += 1



    mse_train = np.mean(mse[:,0])
    mse_test = np.mean(mse[:,1])
    r2_train = np.mean(r2[:,0])
    r2_test = np.mean(r2[:,1])
    b_train = np.mean(b[:,0])
    b_test = np.mean(b[:,1])
    var_train = np.mean(var[:,0])
    var_test = np.mean(var[:,1])


    return mse_train, mse_test, r2_train, r2_test, b_train, b_test, var_train, var_test
