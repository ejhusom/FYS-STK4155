#!/usr/bin/env python3
# ============================================================================
# File:     regression_classification_analysis.py
# Author:   Erik Johannes Husom
# Created:  2019-10-22
# ----------------------------------------------------------------------------
# Description:
# Analyze regression and classification methods.
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as skl
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import sys

from pylearn.resampling import CV
from pylearn.linearmodel import Regression
from pylearn.logisticregression import SGDClassification
from pylearn.metrics import *
from pylearn.neuralnetwork import NeuralNetwork
from pylearn.morten_nn import *

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

    print(CV(X, y, Ridge(alpha=0.00001), n_splits=20,
        classification=False))
    print(CV(X, y, Regression(method='ridge', alpha=0.00001), n_splits=20,
        classification=False))


def logistic_analysis(X, y):

    print(CV(X, y, SGDClassifier(), n_splits=10))       # sklearn
    print(CV(X, y, SGDClassification(), n_splits=10))   # pylearn



def nn_classification(X, y):

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
#    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

    # Reduce size of train sets if necessary
#    X_train = X_train[:10,:]
#    y_train_1hot = y_train_1hot[:10,:]
#    print(np.shape(y_train_1hot))


    hl = [50,50,50]

    # Scikit-learn NN
#    dnn = MLPClassifier(hidden_layer_sizes=hl, activation='logistic',
#                            alpha=0.1, learning_rate_init=0.1, max_iter=1000,
#                            batch_size=100, learning_rate='constant')
#    dnn.fit(X_train, y_train_1hot)
#    print(f'Scikit: {dnn.score(X_test, y_test_1hot)}')

#    etas = np.linspace(0.01, 0.1, 10)
#    etas = [0.1]

#    for eta in etas:
#        print('----------------')
#        print('Eta: {0:.3f}'.format(eta))

    neural = NeuralNetwork(hidden_layer_sizes=hl,
            n_categories=2, 
            eta=0.001, 
            alpha=0.1,
            batch_size=100,
            n_epochs=1000,
            act_func_str='sigmoid',
            output_func_str='sigmoid',
            cost_func_str='crossentropy')

    neural.fit(X_train, y_train_1hot)
    y_pred = neural.predict(X_test)
    print(f'Our code: {accuracy_score(y_test, y_pred)}')



def nn_regression(X, y):
    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    # Scaling
#    sc = StandardScaler()
    sc = MinMaxScaler()
#    X_train = sc.fit_transform(X_train)
#    X_test = sc.transform(X_test)



    hl = [100,20]
    neural = NeuralNetwork(hidden_layer_sizes=hl,
            n_categories=1, eta=0.001, alpha=0.1, batch_size=50,
            n_epochs=100, 
            act_func_str='sigmoid',
            cost_func_str='mse',
            output_func_str='identity')

    neural.fit(X_train, y_train)
    y_pred = neural.predict(X)

#    print(f'Our code R2: {r2_score(y_test, y_pred)}')
#    print(f'Our code MSE: {mean_squared_error(y_test, y_pred)}')

    # Scikit-learn NN
#    y_train = np.ravel(y_train)
#    y_test = np.ravel(y_test)
#    dnn = MLPRegressor(hidden_layer_sizes=hl, 
#        activation='relu',
#        alpha=0.1, 
#        learning_rate_init=0.01, 
#        max_iter=1000,
##       batch_size=100, 
#        tol=1e-7,
#        learning_rate='adaptive')
#    dnn.fit(X_train, y_train)
#    y_pred = dnn.predict(X)

#    print(f'Scikit: {r2_score(y_test, y_pred)}')
#    print(f'Scikit MSE: {mean_squared_error(y_test, y_pred)}')

    n = 101
    x1 = X[:,0].reshape(n,n)
    x2 = X[:,1].reshape(n,n)

    yp = y.reshape(n,n)
    yp_pred = y_pred.reshape(n,n)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, yp, cmap=cm.coolwarm)
    ax.plot_wireframe(x1, x2, yp_pred)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, np.abs(yp-yp_pred))
    plt.show()



if __name__ == '__main__':
    np.random.seed(2020)
    X_b, y_b = create_breast_cancer_dataset()
    X_f, y_f = create_franke_dataset()
#    regression_analysis(X_f, y_f)
#    logistic_analysis(X_b, y_b)
#    nn_classification(X_b, y_b)
    nn_regression(X_f, y_f)
