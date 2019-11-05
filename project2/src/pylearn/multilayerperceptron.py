#!/usr/bin/env python3
# ============================================================================
# File:     multilayerperceptron.py
# Author:   Erik Johannes Husom.
# Created:  2019-10-22
# Version:  0.2
# ----------------------------------------------------------------------------
# DESCRIPTION:
# Feedforward, fully connected, multilayer perceptron artificial neural
# network.
#
# Used for:
# - Regression
# - Binary classification
# ============================================================================
import numpy as np
import time
import sys

class MultilayerPerceptron:
    """Artifical neural network for machine learning purposes, with multilayer
    perceptrons. The number of layers and neurons in each layer is flexible.

    """

    def __init__(
            self,
            hidden_layer_sizes=[50],
            n_epochs=1000,
            batch_size='auto',
            eta=0.1,
            learning_rate='constant',
            alpha=0.0,
            bias0=0.01,
            act_func_str='sigmoid',
            output_func_str='softmax',
            cost_func_str='crossentropy'):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eta = eta/batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.bias0 = bias0
        self.act_func_str = act_func_str
        self.output_func_str = output_func_str
        self.cost_func_str = cost_func_str


    def _initialize(self, X, y):
        self.X_full = X
        self.y_full = y

        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_categories = y.shape[1]

        if self.batch_size == 'auto':
            self.batch_size = min(100, self.n_inputs)
        self.n_batches = int(self.n_inputs/self.batch_size)

        self.layers = ([self.n_features] + 
                        self.hidden_layer_sizes +
                        [self.n_categories])
        self.n_layers = len(self.layers)
        self.set_cost_func(self.cost_func_str)
        self.set_act_func(self.act_func_str)
        self.set_output_func(self.output_func_str)
        self.n_iterations = self.n_inputs // self.batch_size


        self.weights = [None]   # weights for each layer
        self.biases = [None]    # biases for each layer
        self.a = [None]         # output for each layer
        self.z = [None]         # activation for each layer
        self.d = [None]         # error for each layer

        for l in range(1, self.n_layers):
           # self.weights.append(np.random.normal(0.0, pow(self.layers[l],
           #     -0.5), (self.layers[l-1], self.layers[l])))
            # self.weights.append(np.random.randn(self.layers[l-1], self.layers[l])) 
            self.weights.append(np.random.normal(
                                    loc=0.0,
                                    scale=np.sqrt(2./(self.layers[l-1] + self.layers[l])),
                                    size=(self.layers[l-1], self.layers[l])))
            self.biases.append(np.zeros(self.layers[l]) + self.bias0)
            self.z.append(None)
            self.a.append(None)
            self.d.append(None)


    def _feed_forward(self):
        self.a[0] = self.X
        for l in range(1, self.n_layers):
            self.z[l] = self.a[l-1] @ self.weights[l] + self.biases[l]
            self.a[l] = self.act_func(self.z[l])
            
        # Overwriting last output with the chosen output function
        self.a[-1] = self.output_func(self.z[-1])


    def _feed_forward_out(self, X):
        a = X
        for l in range(1, self.n_layers):
            z = a @ self.weights[l] + self.biases[l]
            a = self.act_func(z)
            
        # Overwriting output with chosen output function
        a = self.output_func(z)
        return a


    def _backpropagation(self):

        self.cost = self.cost_func(self.a[-1])

        # NOTE: The following calculation of output error works with the following
        # combination of output activation / loss function:
        # - Sigmoid / binary cross entropy
        # - Softmax / categorical cross entropy
        # - Identity / squared loss
        self.d[-1] = self.a[-1] - self.y

        self.dw = self.a[-2].T @ self.d[-1]
        self.db = np.sum(self.d[-1], axis=0)

        if self.alpha > 0.0:
            self.dw += self.alpha * self.weights[-1]

        self.weights[-1] -= self.eta * self.dw
        self.biases[-1] -= self.eta * self.db


        for l in range(self.n_layers-2, 0, -1):
            self.d[l] = (
                    self.d[l+1] @ self.weights[l+1].T *
                    self.act_func_der(self.z[l])
            )
            
            self.dw = self.a[l-1].T @ self.d[l]

            if self.alpha > 0.0:
                self.dw += self.alpha * self.weights[l]

            self.weights[l] -= self.eta * self.dw
            self.biases[l] -= self.eta * np.sum(self.d[l], axis=0)
            




    def fit(self, X, y):

        self._initialize(X, y)

        if self.learning_rate == 'adaptive':
            t0 = 5
            t1 = 50
            eta = lambda t: 0.01*t0/(t + t1)
        else:
            eta = lambda t: self.eta

        for i in range(self.n_epochs):
            j = 0
            indeces = np.arange(self.n_inputs)
            np.random.shuffle(indeces)
            for batch in range(self.n_batches):
                self.eta = eta(i*self.n_batches + batch)

                rand_indeces = indeces[j*self.batch_size:(j+1)*self.batch_size]
                self.X = self.X_full[rand_indeces, :]
                self.y = self.y_full[rand_indeces]

                self._feed_forward()
                self._backpropagation()

                j += 1
                # print(batch)

            print(f'Epoch {i+1}/{self.n_epochs}. Cost: {self.cost}', end='\r')

        print('\nTraining done.')


    def predict_class(self, X):
        output = self._feed_forward_out(X)
        return np.argmax(output, axis=1)


    def predict(self, X):
        return self._feed_forward_out(X)


    def set_cost_func(self, cost_func_str):

        if cost_func_str == 'mse':
            self.cost_func = self.mse
            self.cost_func_der = self.mse_der
        elif cost_func_str == 'crossentropy':
            self.cost_func = self.crossentropy
            self.cost_func_der = self.crossentropy_der


    def set_act_func(self, act_func_str):
        if act_func_str == 'sigmoid':
            self.act_func = self.sigmoid
            self.act_func_der = self.sigmoid_der
        elif act_func_str == 'tanh':
            self.act_func = self.tanh
            self.act_func_der = self.tanh_der
        elif act_func_str == 'relu':
            self.act_func = self.relu
            self.act_func_der = self.relu_der


    def set_output_func(self, output_func_str='softmax'):

        if output_func_str == 'sigmoid':
            self.output_func = self.sigmoid
            self.output_func_der = self.sigmoid_der
        elif output_func_str == 'softmax':
            self.output_func = self.softmax
            self.output_func_der = self.softmax_der
        elif output_func_str == 'relu':
            self.output_func = self.relu
            self.output_func_der = self.relu_der
        elif self.output_func_str == 'identity':
            self.output_func = lambda x: x
            self.output_func_der = lambda x: 1


    def mse(self, x):
        return 0.5*((x - self.y)**2).mean()

    def mse_der(self, x):
        return x - self.y

    def crossentropy(self, x):
        return - (self.y * np.log(x) + (1 - self.y) * np.log(1 - x)).mean()
#        return -self.y * np.log(x)

    def crossentropy_der(self, x):
        return -self.y/x + (1 - self.y)/(1 - x)
#        return -self.y/x

    def sigmoid(self, x):
        # x[x < -10000] = -1000
        z = 1/(1 + np.exp(-x))
        # print(z)
        return z

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def tanh(self, x):
#        return 2/(1 + np.exp(-2*x)) - 1
        return np.tanh(x)

    def tanh_der(self, x):
        return 1 - self.tanh(x)**2

    def softmax(self, x):
        exp_term = np.exp(x)
        return exp_term / exp_term.sum(axis=1, keepdims=True)


    def relu(self, x):
        return (x >= 0) * x


    def relu_der(self, x):
        return 1. * (x >= 0)


