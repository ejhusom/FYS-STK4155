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

class SGDClassification():

    def __init__(self, eta0=0.1, learning_rate='constant'):

        self.X = None
        self.y = None
        self.y_pred = None
        self.beta = None

        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.batch_size = None



    def predict(self, X=None, beta=None):
        if X is None:
            X = self.X
        if beta is None:
            beta = self.beta

        #y_pred = X @ beta
        #self.y_pred = (y_pred > 0).astype(np.int)
        self.y_pred = self.sigmoid(X, beta)

        return self.y_pred

    def fit(self, X, y, batch_size=10, n_epochs=100):

        self.X = X
        self.y = y

        n = y.size
        n_batches = int(n/batch_size)


        beta = np.random.rand(X.shape[1])

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


if __name__ == '__main__':
    pass
