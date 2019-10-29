#!/usr/bin/env python3
# ============================================================================
# File:     neuralnetwork.py
# Author:   Erik Johannes Husom, based on code example by Morten Hjorth-Jensen.
# Created:  2019-10-22
# Version:  0.1
# ----------------------------------------------------------------------------
# DESCRIPTION:
# Feedforward, fully connected, multilayer perceptron artificial neural
# network.
# ============================================================================
import numpy as np

class NeuralNetwork:
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
            X,
            y,
            hidden_layer_sizes=[50],
            n_categories=2,
            n_epochs=1000,
            batch_size=100,
            eta=0.1,
            alpha=0.1,
            bias0=0.01):

        self.X_full = X
        self.y_full = y

        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        self.hidden_layer_sizes = hidden_layer_sizes
        # TODO: Give error if there are zero hidden layers, or if there are
        # zero neurons in any hidden layers
        self.n_categories = n_categories
        self.layers = [self.n_features] + self.hidden_layer_sizes + [self.n_categories]
        self.n_layers = len(self.layers)

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.alpha = alpha
        self.bias0 = bias0

        self.setup_arrays()
        self.set_cost_func()
        self.set_act_func()
        self.set_output_func()

    def setup_arrays(self):

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
            
#        exp_term = np.exp(self.z[-1])
#        self.a[-1] = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        # Overwriting last output with the chosen output function
        self.a[-1] = self.output_func(self.z[-1])


    def backpropagation(self):

        self.d[-1] = self.a[-1] - self.y

        self.weights_gradient = self.a[-2].T @ self.d[-1]
#        self.bias_gradient = np.sum(self.d[-1], axis=0)

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

    def fit(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.n_epochs):
            for j in range(self.n_iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                self.X = self.X_full[chosen_datapoints]
                self.y = self.y_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


    def set_cost_func(self, cost_func_str='mse'):

        if cost_func_str == 'mse':
            self.cost_func = np.vectorize(lambda y: 0.5*(y - self.y)**2)
            self.cost_func_der = np.vectorize(lambda y: y - self.y)

    def set_act_func(self, act_func_str='sigmoid'):
        # TODO: Implement alternative activation functions.

        if act_func_str == 'sigmoid':
            self.act_func = self.sigmoid
            self.act_func_der = self.sigmoid_der


    def set_output_func(self, output_func_str='softmax'):

        if output_func_str == 'softmax':
            self.output_func = self.softmax
        elif output_func_str == 'sigmoid':
            self.output_func == self.sigmoid


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def softmax(self, x):
        exp_term = np.exp(x)
        return exp_term / np.sum(exp_term, axis=1, keepdims=True)

