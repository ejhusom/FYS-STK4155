#!/usr/bin/env python3
# ============================================================================
# File:     pylearn/activationfunctions.py
# Author:   Erik Johannes Husom
# Created:  2019-10-29
# ----------------------------------------------------------------------------
# Description:
# Collection of activation functions.
# ============================================================================



def sigmoid(self, x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(self, x):
    return self.sigmoid(x)*(1 - self.sigmoid(x))

def softmax(self, x):
    exp_term = np.exp(x)
    return exp_term / np.sum(exp_term, axis=1, keepdims=True)
