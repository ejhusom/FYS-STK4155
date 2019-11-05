#!/usr/bin/env python3
# ============================================================================
# File:     main.py
# Author:   Erik Johannes Husom
# Created:  2019-10-22
# ----------------------------------------------------------------------------
# DESCRIPTION:
# Analyze regression and classification methods.
#
# NOTES:
#
# Pylearn:
# - For regression with relu: eta must be 0.000001
# - For classification: eta must be 0.001
#
# Breast cancer data:
# - MinMaxScaler seems to give better results than StandardScaler.
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import time

import sklearn as skl
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from scikitplot.metrics import plot_confusion_matrix
import scikitplot.metrics as skplt

from pylearn.resampling import CV
from pylearn.linearmodel import Regression
from pylearn.logisticregression import SGDClassification
from pylearn.multilayerperceptron import MultilayerPerceptron

from breastcancer import *
from creditcard import *
from franke import *


def scale_data(train_data, test_data, scaler='standard'):

    if scaler == 'standard':
        sc = StandardScaler()
    elif scaler == 'minmax':
        sc = MinMaxScaler()
    else:
        print('Scaler must be "standard" or "minmax"!')
        return None

    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    return train_data, test_data


def regression_analysis(X, y):

    print(CV(X, y, Ridge(alpha=0.00001), n_splits=20,
        classification=False))  # sklearn
    print(CV(X, y, Regression(method='ridge', alpha=0.00001), n_splits=20,
        classification=False))  # pylearn


def logistic_analysis(X, y):

    print(CV(X, y, SGDClassifier(), n_splits=10))       # sklearn
    print(CV(X, y, SGDClassification(), n_splits=10))   # pylearn



def nn_classification(X, y, scale_columns=None, pl=True, skl=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    if scale_columns is not None:
        minmaxscaler = MinMaxScaler()
        scaler = ColumnTransformer(
                            remainder='passthrough',
                            transformers=[('minmaxscaler', minmaxscaler, scale_columns)])
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


    # One hot encoding targets
    y_train = y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()
    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

    # Reduce size of train sets if necessary
#    X_train = X_train[:10,:]
#    y_train_1hot = y_train_1hot[:10,:]
#    print(np.shape(y_train_1hot))


    hl = [50,50]

    if pl:
        model = MultilayerPerceptron(hidden_layer_sizes=hl,
                eta=0.1, 
                learning_rate='constant',
                alpha=0.0,
                batch_size=100,
                n_epochs=200,
                act_func_str='sigmoid',
                output_func_str='sigmoid',
                cost_func_str='crossentropy')

        model.fit(X_train, y_train_1hot)
        y_pred = model.predict_class(X_test)
        print(f'pylearn accuracy: {accuracy_score(y_test, y_pred)}')
        nn_classification_plot(y_test, y_pred)

    if skl:
        dnn = MLPClassifier(hidden_layer_sizes=hl, activation='logistic',
                                alpha=0.0, learning_rate_init=0.001, max_iter=1000,
                                batch_size=100, learning_rate='constant')
        dnn.fit(X_train, y_train_1hot)
        y_pred = dnn.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print(f'Scikit accuracy: {accuracy_score(y_test, y_pred)}')
        nn_classification_plot(y_test, y_pred)




def nn_classification_plot(y_test, y_pred):

    ax = plot_confusion_matrix(y_test, y_pred, normalize=True, cmap='Blues')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()



def nn_regression(X, y, pl=True, skl=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    hl = [100,100,100,100]
    n_epochs = 1000


    if pl:
        model = MultilayerPerceptron(
                    hidden_layer_sizes=hl,
                    eta=0.01, 
                    alpha=0.000, 
                    batch_size=100,
                    learning_rate='constant',
                    n_epochs=n_epochs, 
                    act_func_str='relu',
                    cost_func_str='mse',
                    output_func_str='identity')

        model.fit(X_train, y_train)
        y_pred = model.predict(X)
        y_pred_test = model.predict(X_test)
        print(f'pylearn R2: {r2_score(y_test, y_pred_test)}')
        print(f'pylearn MSE: {mean_squared_error(y_test, y_pred_test)}')
        nn_regression_plot(X, y, y_pred)

    if skl:
        # Scikit-learn NN
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        dnn = MLPRegressor(
                hidden_layer_sizes=hl, 
                activation='logistic',
                alpha=0.1, 
                learning_rate_init=0.01, 
                max_iter=n_epochs,
                batch_size=200, 
                tol=1e-7,
                learning_rate='constant')

        dnn.fit(X_train, y_train)
        y_pred = dnn.predict(X)
        y_pred_test = dnn.predict(X_test)
        print(f'sklearn R2: {r2_score(y_test, y_pred_test)}')
        print(f'sklearn MSE: {mean_squared_error(y_test, y_pred_test)}')
        nn_regression_plot(X, y, y_pred)



def nn_regression_plot(X, y, y_pred):

    n = int(np.sqrt(np.size(y)))
    x1 = X[:,0].reshape(n,n)
    x2 = X[:,1].reshape(n,n)
    y_mesh = y.reshape(n,n)
    y_pred_mesh = y_pred.reshape(n,n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, y_mesh, cmap=cm.coolwarm)
    ax.plot_wireframe(x1, x2, y_pred_mesh)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, np.abs(y_mesh - y_pred_mesh))
    plt.show()



if __name__ == '__main__':
    np.random.seed(2010)

    # Create data sets
    X_b, y_b = breast_cancer_dataset()
    franke = FrankeDataset(n=100, eps=0.01)
    X_f, y_f = franke.generate_data_set()
    # X_c, y_c, scale_columns = preprocess_creditcard_data('../data/credit_card.xls')

    # Analyze data
#    regression_analysis(X_f, y_f)
#    logistic_analysis(X_b, y_b)
#    nn_classification(X_b, y_b)
    # nn_classification(X_c, y_c, scale_columns, pl=True, skl=True)
    nn_regression(X_f, y_f, pl=True, skl=False)
