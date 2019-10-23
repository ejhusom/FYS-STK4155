#!/usr/bin/env python3
# ============================================================================
# File:     gradientdescent.py
# Author:   Erik Johannes Husom
# Created:  2019-10-15
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
import numpy as np
from sklearn.model_selection import KFold

class SGDClassification():

    def __init__(self, eta0=0.1, learning_rate='constant'):

        self.X = None
        self.y = None
        self.y_pred = None
        self.beta = None

        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.batch_size = None



    def predict(self, X=None, beta=None, probability=False):
        if X is None:
            X = self.X
        if beta is None:
            beta = self.beta

        self.y_pred = self.sigmoid(X, beta)

        if probability==False:
            self.y_pred = (self.y_pred > 0.5).astype(np.int)

        return self.y_pred

    def fit(self, X, y, batch_size=100, n_epochs=1000):

        self.X = X
        self.y = y

        n = y.size
        n_batches = int(n/batch_size)


        beta = np.random.randn(X.shape[1])

        for epoch in range(n_epochs):
            indeces = np.arange(n)
            np.random.shuffle(indeces)
            j = 0
            for batch in range(n_batches):
                rand_indeces = indeces[j*batch_size:(j+1)*batch_size]
                X_i = X[rand_indeces, :]
                y_i = y[rand_indeces]
                
                gradients = X_i.T @ (self.sigmoid(X_i, beta) - y_i)
                beta -= self.eta0*gradients
                j += 1

        self.beta = beta
        return beta



    def learning_schedule(self, t, t0=1):
        return 1.0/(t + t0)


    def sigmoid(self, X, beta):

        term = np.exp(X @ beta)
        return term / (1 + term)


#    def CV(self, X, y, n_splits=5):
#
#        self.X = X
#        self.y = y
#
#        kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
#
#        mse = np.zeros(n_splits)
#        r2 = np.zeros(n_folds)
#
#        i = 0
#        for train_idx, val_idx in kf-split(self.X):
#            self.fit(self.X[train_idx], self.y[train_idx])
#            self.predict(self.X[val_idx])
#            mse[i] = mean_squared_error(self.y[val_idx], self.y_pred)
#            r2[i] = r2_score(self.y[val_idx], self.y_pred)
#
#            i += 1
#
#        self.mse_cv = np.mean(mse)
#        self.r2_cv = np.mean(r2)
#
#        return self.mse_cv, self.r2_cv






if __name__ == '__main__':
    pass
