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

class GradientDescent():

    def __init__(self, mode='classification', eta0=0.1, learning_rate='constant'):

        self.X = None
        self.y = None
        self.y_pred = None
        self.beta = None

        self.mode = mode
        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.batch_size = None



    def predict(self, X=None, beta=None):
        if X is None:
            X = self.X
        if beta is None:
            beta = self.beta

        self.y_pred = X @ beta

        return self.y_pred


    def SGD(self, X, y, batch_size=5, n_epochs=100):

        n = y.size
        n_batches = int(n/batch_size)


        if self.mode=='regression':
            beta = np.random.randn(X.shape[1],1)
        else:
            beta = np.random.rand(X.shape[1])

        for epoch in range(n_epochs):
            indeces = np.arange(n)
            np.random.shuffle(indeces)
            j = 0
            for batch in range(n_batches):
                rand_indeces = indeces[j*batch_size:(j+1)*batch_size]
                X_i = X[rand_indeces, :]
                y_i = y[rand_indeces]
                

                if self.mode=='regression':
                    gradients = self.squared_loss(X_i, y_i, beta)
                else:
                    gradients = X_i.T @ (self.sigmoid(X_i, beta) - y_i)
                beta -= self.eta0*gradients
                j += 1

        return beta

    def GD(self, X, y, n_iter=1000):
      
        beta = np.random.randn(X.shape[1],1)

        for iter in range(n_iter):
            if self.mode=='regression':
                gradients = self.squared_loss(X, y, beta)
            else:
                gradients = X.T @ (self.sigmoid(X, beta) - y)
            beta -= self.eta0*gradients

        self.beta = beta
        return self.beta


    def learning_schedule(self, t, t0=1):
        return 1.0/(t + t0)


    def sigmoid(self, X, beta):

        if len(np.shape(beta)) > 1:
            beta = np.ravel(beta)

        expXbeta = np.exp(X @ beta)
        return expXbeta / (1 + expXbeta)


    def squared_loss(self, X, y, beta):

        n = y.size
        return 2.0/n*X.T @ ((X @ beta) - y)





if __name__ == '__main__':
    pass
