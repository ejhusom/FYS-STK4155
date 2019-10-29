#!/usr/bin/env python3
# ============================================================================
# File:     breast_cancer_analysis.py
# Author:   Erik Johannes Husom
# Created:  2019-10-22
# ----------------------------------------------------------------------------
# Description:
# Analyze breast cancer data.
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import sys

from pylearn.resampling import CV
from pylearn.logisticregression import SGDClassification
from pylearn.neuralnetwork import NeuralNetwork
from pylearn.morten_nn import NeuralNetwork_M

from franke import *

def visualize(df):
    plt.figure(figsize=(10,10))
    features = list(df.columns[1:10])
    ax = sns.heatmap(df[features].corr(), square=True, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

def sklearn_bunch_to_pandas_df(bunch):
    data = np.c_[bunch.data, bunch.target]
    columns = np.append(bunch.feature_names, ['target'])
    return pd.DataFrame(data, columns=columns)

def create_breast_cancer_dataset():
    # Reading data
    data = load_breast_cancer()
    y_names = data['target_names']
    y = data['target']
    X_names = data['feature_names']
    X = data['data']

    return X, y


def regression_analysis(X, y):

    print(CV(X, y, Regression(method='ridge', alpha=0.00001), n_splits=20, classification=False))

    # Old code without cross-validation
#    X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#    sc = StandardScaler()
#    X_train = sc.fit_transform(X_train)
#    X_test = sc.transform(X_test)
#
#    model = Regression()
#    model.fit(X_train, y_train)
#    model.predict(X_test)
#    print(mean_squared_error(model.y_pred, y_test))
#    print(r2_score(y_test, model.y_pred))


def logistic_analysis(X, y):

    print(CV(X, y, SGDClassifier(), n_splits=10))       # sklearn
    print(CV(X, y, SGDClassification(), n_splits=10))   # pylearn



def neural_network_analysis(X, y):

    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    # Scaling
    # NOTE: MinMaxScaler seems to give better results than StandardScaler on
    # breast cancer data.
#    sc = StandardScaler()
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # One hot encoding targets
    y_train= y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()
    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

    # Reduce size of train sets if necessary
#    X_train = X_train[:10,:]
#    y_train_1hot = y_train_1hot[:10,:]
#    print(np.shape(y_train_1hot))


    hl = [50]

    # Scikit-learn NN
#    dnn = MLPClassifier(hidden_layer_sizes=hl, activation='logistic',
#                            alpha=0.1, learning_rate_init=0.1, max_iter=1000,
#                            batch_size=100, learning_rate='constant')
#    dnn.fit(X_train, y_train_1hot)
#    print(f'Scikit: {dnn.score(X_test, y_test_1hot)}')

    etas = np.linspace(0.01, 0.1, 10)
    etas = [0.001]

    for eta in etas:
        print('----------------')
        print('Eta: {0:.3f}'.format(eta))

        neural = NeuralNetwork(X_train, y_train_1hot, hidden_layer_sizes=hl,
                n_categories=2, eta=eta, alpha=0.1, batch_size=10,
                n_epochs=100)

        neural.fit()
        y_pred = neural.predict(X_test)
        print(f'Our code: {accuracy_score(y_test, y_pred)}')


if __name__ == '__main__':
    np.random.seed(2020)
    X_b, y_b = create_breast_cancer_dataset()
    X_f, y_f = create_franke_dataset()
#    regression_analysis(X_f, y_f)
#    logistic_analysis(X_b, y_b)
    neural_network_analysis(X_b, y_b)
