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



def preprocessing_breast_cancer():
    # Reading data
    data = load_breast_cancer()
    y_names = data['target_names']
    y = data['target']
    X_names = data['feature_names']
    X = data['data']

    return X, y


def logistic_analysis(X, y):

    # Cross-validation
    print(CV(X, y, SGDClassifier(), n_splits=10))
    print(CV(X, y, SGDClassification(), n_splits=10))



def neural_network_analysis(X, y, single):

    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    # Standardizing
#    sc = StandardScaler()
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # One hot encoding targets
    y_train= y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()

    encoder = OneHotEncoder(categories='auto')
    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

    # Reduce size of train sets if necessary
#    X_train = X_train[:10,:]
#    y_train_1hot = y_train_1hot[:10,:]
#    print(np.shape(y_train_1hot))

    # Scikit-learn NN
    dnn = MLPClassifier(hidden_layer_sizes=(50,50), activation='logistic',
                            alpha=0.1, learning_rate_init=0.1, max_iter=1000)
    dnn.fit(X_train, y_train_1hot)
    print(f'Scikit: {dnn.score(X_test, y_test_1hot)}')



#    neural = NeuralNetwork(X_train, y_train_1hot, hidden_layers=[50],
#            n_categories=2, single=single, alpha=0.1, batch_size=50,
#            n_epochs=2000)

#    neural.train()
#    y_pred = neural.predict(X_test)
#    print(accuracy_score(y_test, y_pred))


    # Morten's NN code
    neural = NeuralNetwork_M(X_train, y_train_1hot, n_hidden_neurons=50,
            n_categories=2, lmbd=0.1, eta=0.1, batch_size=100,
            epochs=1000)

    neural.train()
    y_pred = neural.predict(X_test)
    print(f'Morten: {accuracy_score(y_test, y_pred)}')


if __name__ == '__main__':
    np.random.seed(2020)
    X, y = preprocessing_breast_cancer()
#    logistic_analysis(X, y)

    if len(sys.argv) > 1:
        single = True
        print('single=True')
    else:
        single = False

    neural_network_analysis(X, y, single=single)
