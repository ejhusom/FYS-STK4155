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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import sys

# Add machine learning methods to path
sys.path.append('./methods')

from crossvalidation import CV
from logisticregression import SGDClassification
from neuralnetwork import NeuralNetwork



def scale(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

#    X_train = X_train - np.mean(X_train, axis=0)
#    X_test = X_test - np.mean(X_train, axis=0)
#    col_std = np.std(X_train, axis=0)
#
#    for i in range(0, np.shape(X_train)[1]):
#        X_train[:,i] /= col_std[i]
#        X_test[:,i] /= col_std[i]


    return X_train, X_test


def visualize(df):

    plt.figure(figsize=(10,10))

    features = list(df.columns[1:10])

    ax = sns.heatmap(df[features].corr(), square=True, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

 



def bunch2dataframe(bunch):

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

    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    # Normalizing data
    X_train, X_test = scale(X_train, X_test)

    sc = StandardScaler()
    X = sc.fit_transform(X)

    return X, y, X_train, X_test, y_train, y_test


def logistic_breast_cancer(X, y, X_train, X_test, y_train, y_test):

    # Prediction with Scikit-learn
    clf_skl = SGDClassifier()
    clf_skl.fit(X_train, y_train)
    y_pred_skl = clf_skl.predict(X_test)
    score_skl = accuracy_score(y_test, y_pred_skl)
    print(score_skl)


    # Cross-validation
    print(CV(X, y, SGDClassifier(), n_splits=10))
    print(CV(X, y, SGDClassification(eta0=0.01), n_splits=10))



def neural_network_analysis(X_train, X_test, y_train, y_test):

    # Creating neural network
    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)
    X_train, X_test = scale(X_train, X_test)

    y_train= y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()

    neural = NeuralNetwork(X_train, y_train_1hot, n_hidden_neurons=10,
            n_categories=2)

    neural.train()
    test_predict = neural.predict(X_test)

    print(accuracy_score(y_test, test_predict))



if __name__ == '__main__':
    np.random.seed(2019)
    X, y, X_train, X_test, y_train, y_test = preprocessing_breast_cancer()
    logistic_breast_cancer(X, y, X_train, X_test, y_train, y_test)
