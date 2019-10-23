#!/usr/bin/env python3
# ============================================================================
# File:     test_gradientdescent.py
# Author:   Erik Johannes Husom
# Created:  2019-10-15
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
import numpy as np
import sys

sys.path.append('./../')
from logisticregression import SGDClassification



def test_SGD_classification(plot=False):
    from sklearn import datasets, linear_model
    import matplotlib.pyplot as plt

    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)

    # Scikit-learn prediction
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    print(clf.coef_)

    pred_func = lambda x: clf.predict(x)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z_skl = Z.reshape(xx.shape)

    # Manual prediction
    model = SGDClassification()
    beta_SGD = model.fit(X, y, batch_size=10)
    print(beta_SGD)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()], beta_SGD)
    Z = Z.reshape(xx.shape)

    diff = np.abs(Z - Z_skl)
    n_errors = abs(np.sum(diff))
    rel_error = n_errors/diff.size
    print(f'No. of erranous classifications: {n_errors}')
    print(f'Percentage of erranous classifications: {100*rel_error}')

    if plot==True:
        fig = plt.figure()
        plt.contourf(xx, yy, Z_skl, cmap=plt.cm.Spectral)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.show()


if __name__ == '__main__':
    np.random.seed(2019)
    test_SGD_classification(plot=True)
