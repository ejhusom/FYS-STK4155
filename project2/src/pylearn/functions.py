#!/usr/bin/env python3
# ============================================================================
# File:     functions.py
# Author:   Erik Johannes Husom
# Created:  2019-11-01
# ----------------------------------------------------------------------------
# Description:
# Mathematical functions used in machine learning.
# ============================================================================
import numpy as np



def mse(x):
    return 0.5*((x - y)**2).mean()

def mse_der(x):
    return x - y

def crossentropy(x):
    return - (y * np.log(x) + (1 - y) * np.log(1 - x)).mean()
#        return -y * np.log(x)

def crossentropy_der(x):
    return -y/x + (1 - y)/(1 - x)
#        return -y/x


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1 - sigmoid(x))

def tanh(x):
#        return 2/(1 + np.exp(-2*x)) - 1
    return np.tanh(x)

def tanh_der(x):
    return 1 - tanh(x)**2

def softmax(x):
    exp_term = np.exp(x)
    return exp_term / exp_term.sum(axis=1, keepdims=True)

def softmax_der(x):
    # FIXME: This does not work.
#        pass
#        s = softmax(x).reshape(-1,1)
#        return np.diagflat(s) - np.dot(s, s.T)
    return sigmoid(x)*(1 - sigmoid(x))

def relu(x):
    return (x >= 0) * x


def relu_der(x):
    return 1. * (x >= 0)


