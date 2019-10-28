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
# 
# NOTES:
# Current main problem is trouble with implementing multilayer functionality.
# The forward feed and backpropagation runs, but gives wrong results, and there
# seems to be something wrong with the signal flow, but uncertain whether it is
# in feed_forward or in backpropagation (or both).
#
# The boolean "single" and the single hidden layer code should be removed once
# the MLP is working.
#
# PLAN:
# - Print weights and/or z for individual layers for single-layer and MLP, and
#   compare in order to find the erranous calculations.
#
# TODO:
# Implement flexible choice of activation function?
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
        hidden_layers : array-like
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
        cost_func :
        cost_func_der :

    Methods
    -------
        create_biases_and_weights
        feed_forward
        feed_forward_out
        backpropagation
        predict
        predict_probabilities
        train
        set_cost_func
        sigmoid
        tanh


    """

    def __init__(
            self,
            X,
            y,
            hidden_layers=[50],
            n_categories=2,
            n_epochs=1000,
            batch_size=100,
            eta=0.1,
            alpha=0.0,
            bias0=0.01,
            single=True):

        self.X_full = X
        self.y_full = y

        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        self.hidden_layers = hidden_layers
        # NOTE: n_hidden_neurons will be deprecated when single layer structure is
        # removed
        self.n_hidden_neurons = hidden_layers[0]
        # TODO: Give error if there are zero hidden layers, or if there are
        # zero neurons in any hidden layers
        self.n_categories = n_categories
        self.layers = [self.n_features] + self.hidden_layers + [self.n_categories]
        self.n_layers = len(self.layers)

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.alpha = alpha
        self.bias0 = bias0

        self.single = single

        self.create_biases_and_weights()
        self.set_cost_func()

    def create_biases_and_weights(self):

        if self.single:
            self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
            self.hidden_bias = np.zeros(self.n_hidden_neurons) + self.bias0
            self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
            self.output_bias = np.zeros(self.n_categories) + self.bias0
        else:
            self.weights = [None]
            self.biases = [None]
            self.a = [None]

            for layer in range(1, self.n_layers):
                self.weights.append(np.random.randn(self.layers[layer-1],
                    self.layers[layer]))
                self.biases.append(np.zeros(self.layers[layer]) +
                        self.bias0)
                self.a.append(None)


    def feed_forward(self):

        if self.single:
            self.z_h = np.matmul(self.X, self.hidden_weights) + self.hidden_bias
            self.a_h = self.sigmoid(self.z_h)
            self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
            exp_term = np.exp(self.z_o)
            self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        else:
            self.a[0] = self.X
            for l in range(1, self.n_layers):
                self.z = (self.a[l-1] @ self.weights[l] + self.biases[l])
                self.a[l] = self.sigmoid(self.z)
                
            exp_term = np.exp(self.z)
            self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
#            self.probabilities = self.a[l]


    def feed_forward_out(self, X):

        if self.single:
            z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
            a_h = self.sigmoid(z_h)

            z_o = np.matmul(a_h, self.output_weights) + self.output_bias
            
            # Uses softmax for output
            exp_term = np.exp(z_o)
            probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
            return probabilities
        else:
            a = X
            for l in range(1, self.n_layers):
                z = a @ self.weights[l] + self.biases[l]
                a = self.sigmoid(z)
                
                
            exp_term = np.exp(z)
            probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
            return probabilities


    def backpropagation(self):

        if self.single:
            error_output = self.probabilities - self.y
            error_hidden = error_output @ self.output_weights.T * self.a_h * (1 - self.a_h)

            self.output_weights_gradient = self.a_h.T @ error_output
            self.output_bias_gradient = np.sum(error_output, axis=0)

            self.hidden_weights_gradient = self.X.T @ error_hidden
            self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

            if self.alpha > 0.0:
                self.output_weights_gradient += self.alpha * self.output_weights
                self.hidden_weights_gradient += self.alpha * self.hidden_weights

            self.output_weights -= self.eta * self.output_weights_gradient
            self.output_bias -= self.eta * self.output_bias_gradient
            self.hidden_weights -= self.eta * self.hidden_weights_gradient
            self.hidden_bias -= self.eta * self.hidden_bias_gradient
#            print(error_hidden)            
#            print(self.hidden_weights)

        else:
            # FIXME: The calculation of errors is wrong, and causes the
            # algorithm to give wrong results.

            # Output layer error and gradients
            error = self.probabilities - self.y
            delta = self.a[-1] - self.y
            # Using self.a_h[-2], because it is the last valid activation layer
            self.weights_gradient = self.a[-2].T @ error
            self.bias_gradient = np.sum(error, axis=0)

            if self.alpha > 0.0:
                self.weights_gradient += self.alpha * self.weights[-1]

            self.weights[-1] -= self.eta * self.weights_gradient
            self.biases[-1] -= self.eta * self.bias_gradient

            # Hidden layers error and gradients
            for l in range(self.n_layers-2, 0, -1):
                error = [
                    error @ self.weights[l+1].T * self.a[l] * (1 - self.a[l])
                    
                ]
#                print(error)

                self.weights_gradient = self.a[l-1].T @ error
                self.bias_gradient = np.sum(error, axis=0)

                if self.alpha > 0.0:
                    self.weights_gradient += (self.alpha * self.weights[l])

                self.weights[l] -= self.eta * self.weights_gradient
                self.biases[l] -= self.eta * self.bias_gradient
            
#            print(self.weights[1])


    def predict(self, X):
        # TODO: Implement other functions than softmax?

        probabilities = self.feed_forward_out(X)
        # Using softmax for prediction
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
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

    def set_act_func(self):

        sigmoid = lambda x: 1/(1 + np.exp(-x))
        self.act_func = np.vectorize(sigmoid)
        self.act_func_der = np.vectorize(lambda x: sigmoid(x)*(1 - sigmoid(x)))


    def set_output_func(self):
        # TODO: Possible to choose a different activation function for the
        # output layer.
        pass


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_der(self, x):
#        return np.exp(-x)/(np.exp(-x) + 1)**2
        return self.sigmoid(x)*(1 - self.sigmoid(x))


    def tanh(x):
        return np.tanh(x)


