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
# ============================================================================
import numpy as np
import sys

class MultilayerPerceptron:
    """Artifical neural network for machine learning purposes, with multilayer
    perceptrons. The number of layers and neurons in each layer is flexible.

    Attributes
    ----------
        X_full :
        X :
        y_full :
        y :
        n_inputs :
        n_features :
        hidden_layer_sizes : array-like
        n_hidden_neurons : 
        n_categories : 
        n_epochs :
        batch_size :
        n_iterations : 
        eta :
        alpha :
        bias0 : float, default=0.01
            Initial bias value for all layers.
        biases : list, containing arrays
        weights : list, containing arrays
        a : list, containing arrays
            Each array in this list contains the values corresponding to each
            neuron in a layer, after the signal has been run through the
            activation function. The arrays are contained in a list because
            they usually are of different lengths. The first item in the list
            contains the input features, because this makes the for-loops
            smooth. The last item in the list is empty, because there are no
            values needed for the output layer.
        z :
        probabilities :

    Methods
    -------
        setup_arrays
        feed_forward
        feed_forward_out
        backpropagation
        predict
        predict_probabilities
        fit
        set_cost_func
        set_act_func
        set_output_func
        sigmoid
        sigmoid_der
        softmax


    """

    def __init__(
            self,
            hidden_layer_sizes=[50],
            n_categories=1,
            n_epochs=1000,
            batch_size='auto',
            eta=0.01,
            learning_rate='constant',
            alpha=0.1,
            bias0=0.01,
            act_func_str='sigmoid',
            output_func_str='softmax',
            cost_func_str='mse'):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_categories = n_categories
        self.batch_size = batch_size

        self.n_epochs = n_epochs
        self.eta = eta
        # TODO: Implement adaptive learning rate
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.bias0 = bias0
        self.act_func_str = act_func_str
        self.output_func_str = output_func_str
        self.cost_func_str = cost_func_str


    def initialize(self, X, y):
        self.X_full = X
        self.y_full = y

        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        if self.batch_size == 'auto':
            self.batch_size = min(100, self.n_inputs)
        self.n_batches = int(self.n_inputs/self.batch_size)

        self.layers = [self.n_features] + self.hidden_layer_sizes + [self.n_categories]
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
            self.weights.append(np.random.randn(self.layers[l-1],
                self.layers[l])) 
            self.biases.append(np.zeros(self.layers[l]) + self.bias0)
            self.z.append(None)
            self.a.append(None)
            self.d.append(None)


    def feed_forward(self):

        self.a[0] = self.X
        for l in range(1, self.n_layers):
            self.z[l] = self.a[l-1] @ self.weights[l] + self.biases[l]
            self.a[l] = self.act_func(self.z[l])
            print(self.z[l])
            
        # Overwriting last output with the chosen output function
        self.a[-1] = self.output_func(self.z[-1])
        sys.exit(1)


    def backpropagation(self):


        self.cost = self.cost_func(self.a[-1])


        # The following calculation of output error works with the following
        # combination of output activation / loss function:
        # - Sigmoid / binary cross entropy
        # - Softmax / categorical cross entropy
        # - Identity / squared loss
        self.d[-1] = self.a[-1] - self.y

#        self.dw = self.a[-2].T @ self.d[-1]
#        self.db = np.sum(self.d[-1], axis=0)

        self.weights[-1] -= self.eta * self.a[-2].T @ self.d[-1]
        self.biases[-1] -= self.eta * np.sum(self.d[-1], axis=0)


        for l in range(self.n_layers-2, 0, -1):
            self.d[l] = (
                    self.d[l+1] @ self.weights[l+1].T *
                    self.act_func_der(self.z[l])
            )

            self.weights[l] -= self.eta * self.a[l-1].T @ self.d[l]
            self.biases[l] -= self.eta * np.sum(self.d[l], axis=0)
        


    def feed_forward_out(self, X):

        a = X
        for l in range(1, self.n_layers):
            z = a @ self.weights[l] + self.biases[l]
            a = self.act_func(z)
            
        # Overwriting output with chosen output function
        a = self.output_func(z)
        return a

    def predict(self, X):
        # TODO: Implement other functions than softmax?
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def fit(self, X, y):

        self.initialize(X, y)

        data_indices = np.arange(self.n_inputs)
        
        cost_limit = 1000
        tol = 1e-7
        cost_decrease_fails = -1

#        self.n_iterations = 1
        for i in range(self.n_epochs):
#            for j in range(self.n_iterations):
#                chosen_datapoints = np.random.choice(
#                    data_indices, size=self.batch_size, replace=False
#                )
            j = 0
            indeces = np.arange(self.n_inputs)
            np.random.shuffle(indeces)
            for batch in range(self.n_batches):
                rand_indeces = indeces[j*self.batch_size:(j+1)*self.batch_size]
                self.X = self.X_full[rand_indeces, :]
                self.y = self.y_full[rand_indeces]

                j += 1
#                self.X = self.X_full[chosen_datapoints]
#                self.y = self.y_full[chosen_datapoints]


                self.feed_forward()
                self.backpropagation()
#                print(f'Cost in iterations: {self.cost}')

            
            # Adapative learning rate: If the cost is not reduced by a minimum
            # of tol in two epochs, the learning rate will be divided by 5.
            print(f'Cost: {self.cost}')
#            if (cost_limit - self.cost) < tol:
#                cost_decrease_fails += 1
#            if cost_decrease_fails > 5:
#                self.eta /= 2.0
#                cost_decrease_fails = 0
#                print(f'Learning rate apapted: {self.eta}')
#
#            cost_limit = self.cost



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
        return 1/(1 + np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def tanh(self, x):
        return 2/(1 + np.exp(-2*x)) - 1

    def tanh_der(self, x):
        return 1 - self.tanh(x)**2

    def softmax(self, x):
        exp_term = np.exp(x)
        return exp_term / exp_term.sum(axis=1, keepdims=True)

    def softmax_der(self, x):
        # FIXME: This does not work.
#        pass
#        s = self.softmax(x).reshape(-1,1)
#        return np.diagflat(s) - np.dot(s, s.T)
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def relu(self, x):
#        z = np.zeros_like(x)
#         return np.clip(x, 0, np.finfo(x.dtype).max, out=None)
#        return z
#        result = 0 if x < 0.0 else x
#        return result
        return np.maximum(0, x)
#        return (x > 0) * x
#        z[x>0]=x
#        z[x<=0]=0
#        return z
#        f = np.vectorize(lambda x: 0 if x < 0.0 else x)
#        return f(x)

#    def relu_der(self, x, alpha=0.01):
#      dx = np.ones_like(x)
#      dx[x < 0] = alpha
#      return dx

    def relu_der(self, x):
#        z = np.zeros_like(x)
#        return np.clip(1, 0, np.finfo(x.dtype).max, out=None)
#        return z
#        return 1. * (x > 0)
#        result = 0 if x < 0.0 else 1
#        return result
#        return (x > 0) * 1
#        return (x > 0).astype(int)
        z = np.zeros_like(x)
        z[x>0]=1
        z[x<=0]=0
#        print(z)
        return z
#        f = np.vectorize(lambda x: 0 if x < 0.0 else 1)
#        return f(x)
