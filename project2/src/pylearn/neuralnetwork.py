#!/usr/bin/env python3
# ============================================================================
# File:     neuralnetwork.py
# Author:   Erik Johannes Husom, based on code example by Morten Hjorth-Jensen.
# Created:  2019-10-22
# ----------------------------------------------------------------------------
# Description:
# Neural network.
# ============================================================================
import numpy as np

class NeuralNetwork:
    """Artifical neural network for machine learning purposes, with multilayer
    perceptrons. The number of layers and neurons in each layer is flexible.

    Attributes
    ----------
        X_full :
        y_full :
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
        bias :

    Methods
    -------
        create_biases_and_weights
        feed_forward
        feed_forward_out
        backpropagation
        predict
        predict_probabilities
        train
        sigmoid
        tanh


    """

    def __init__(
            self,
            X,
            y,
            hidden_layers=[50],
            n_categories=10,
            n_epochs=100,
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
        self.n_hidden_neurons = hidden_layers[0]
        self.n_categories = n_categories

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.alpha = alpha
        self.bias0 = bias0

        self.single = single

        self.create_biases_and_weights()

    def create_biases_and_weights(self):

        # Creating lists containing weights and biases for all layers
        self.hidden_weights = []
        self.hidden_bias = []



        if self.single:
            self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
            self.hidden_bias = np.zeros(self.n_hidden_neurons) + self.bias0
            self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
            self.output_bias = np.zeros(self.n_categories) + self.bias0
        else:

            self.weights = []
            self.bias = []

            for layer in range(len(self.hidden_layers)):
                self.weights.append(np.random.randn(self.n_features,
                    self.hidden_layers[layer]))

                self.bias.append(np.zeros(self.hidden_layers[layer]) +
                        self.bias0)


            self.weights.append(np.random.randn(self.hidden_layers[-1],
                self.n_categories))
            self.bias.append(np.zeros(self.n_categories) + self.bias0)


#        print(np.shape(self.hidden_weights))
#        print(np.shape(self.hidden_bias))
#        print(np.shape(self.output_weights))
#        print(np.shape(self.output_bias))


    def feed_forward(self):


        if self.single:
            self.z_h = np.matmul(self.X, self.hidden_weights) + self.hidden_bias
            self.a_h = self.sigmoid(self.z_h)
            self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
            exp_term = np.exp(self.z_o)
            self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        else:

            for layer in range(len(self.hidden_layers)):
                self.z_h = (self.X @ self.weights[layer] 
                            + self.bias[layer])
                self.a_h = self.sigmoid(self.z_h)
                
            self.z_o = self.a_h @ self.weights[-1] + self.bias[-1]
            exp_term = np.exp(self.z_o)
            self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)


    def feed_forward_out(self, X):

        if self.single:
            z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
            a_h = self.sigmoid(z_h)

            z_o = np.matmul(a_h, self.output_weights) + self.output_bias
            
            exp_term = np.exp(z_o)
            probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
            return probabilities
        else:
            z_h = np.matmul(X, self.hidden_weights[-1]) + self.hidden_bias[-1]
            a_h = self.sigmoid(z_h)

            z_o = np.matmul(a_h, self.output_weights) + self.output_bias
            
            exp_term = np.exp(z_o)
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
        else:

            # Output layer error and gradients
            error_output = self.probabilities - self.y
            self.output_weights_gradient = self.a_h.T @ error_output
            self.output_bias_gradient = np.sum(error_output, axis=0)

            if self.alpha > 0.0:
                self.output_weights_gradient += self.alpha * self.output_weights

            self.output_weights -= self.eta * self.output_weights_gradient
            self.output_bias -= self.eta * self.output_bias_gradient

            # Hidden layers error and gradients
            error_hidden = []
            for layer in range(len(self.hidden_layers)):
                error_hidden.append(
                        error_output @ self.output_weights.T * self.a_h 
                        * (1 - self.a_h)
                        )


                self.hidden_weights_gradient = self.X.T @ error_hidden[layer]
                self.hidden_bias_gradient = np.sum(error_hidden[layer], axis=0)

                if self.alpha > 0.0:
                    self.hidden_weights_gradient += (self.alpha *
                                                    self.hidden_weights[layer])

                self.hidden_weights[layer] -= self.eta * self.hidden_weights_gradient
                self.hidden_bias[layer] -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
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

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)


if __name__ == '__main__':
    pass
