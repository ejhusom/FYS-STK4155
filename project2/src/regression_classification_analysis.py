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
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import sys
import time
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix
import scikitplot.metrics as skplt
import seaborn as sns
from pylearn.resampling import CV
from pylearn.linearmodel import Regression
from pylearn.logisticregression import SGDClassification
from pylearn.metrics import *
from pylearn.multilayerperceptron import MultilayerPerceptron

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

def scale_data(train_data, test_data, scaler='standard'):

    if scaler == 'standard':
        sc = StandardScaler()
    else:
        sc = MinMaxScaler()

    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    return train_data, test_data

def regression_analysis(X, y):

    print(CV(X, y, Ridge(alpha=0.00001), n_splits=20,
        classification=False))
    print(CV(X, y, Regression(method='ridge', alpha=0.00001), n_splits=20,
        classification=False))


def logistic_analysis(X, y):

    print(CV(X, y, SGDClassifier(), n_splits=10))       # sklearn
    print(CV(X, y, SGDClassification(), n_splits=10))   # pylearn


def preprocess_CC_data(filename):

    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)


    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


    #find and remove outliers in the data
    outlier_gender1 = np.where(X[:,1] < 1)[0]
    outlier_gender2 = np.where(X[:,1] > 2)[0]

    outlier_education1 = np.where(X[:,2] < 1)[0]
    outlier_education2 = np.where(X[:,2] > 4)[0]

    outlier_marital1 = np.where(X[:,3] < 1)[0]
    outlier_marital2 = np.where(X[:,3] > 3)[0]

    inds = np.concatenate((outlier_gender1,
                           outlier_gender2,
                           outlier_education1,
                           outlier_education2,
                           outlier_marital1,
                           outlier_marital2))


    outlier_rows = np.unique(inds)

    X = np.delete(X, outlier_rows, axis=0)
    y = np.delete(y, outlier_rows, axis=0)


    #split data into categorical and continuous features
    categorical_inds = (1, 2, 3)
    continuous_inds = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    X_cat = X[:,categorical_inds]
    X_cont = X[:, continuous_inds]


    #onehot encode categorical data
    onehotencoder = OneHotEncoder(categories="auto", sparse=False)
    preprocessor = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                ('onehot', onehotencoder, list(range(X_cat.shape[1])))])

    X_cat = preprocessor.fit_transform(X_cat)

    #join categorical and continuous features
    X = np.concatenate((X_cont, X_cat), axis=1)

    cont_feat_inds = list(range(X_cont.shape[1]))

    return X, np.ravel(y), cont_feat_inds



def nn_classification(X, y, scale_columns=None):

    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    # Scaling
    # NOTE: MinMaxScaler seems to give better results than StandardScaler on
    # breast cancer data.
#    sc = StandardScaler()
#    sc = MinMaxScaler()
#    X_train = sc.fit_transform(X_train)
#    X_test = sc.transform(X_test)
    # CC
    minmaxscaler = MinMaxScaler()
    scaler = ColumnTransformer(
                        remainder='passthrough',
                        transformers=[('minmaxscaler', minmaxscaler, scale_columns)])
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # One hot encoding targets
    y_train= y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()
    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

    # Reduce size of train sets if necessary
#    X_train = X_train[:10,:]
#    y_train_1hot = y_train_1hot[:10,:]
#    print(np.shape(y_train_1hot))


    hl = [50,50,50]

    # Scikit-learn NN
    dnn = MLPClassifier(hidden_layer_sizes=hl, activation='logistic',
                            alpha=0.0, learning_rate_init=0.001, max_iter=1000,
                            batch_size=100, learning_rate='constant')
    dnn.fit(X_train, y_train_1hot)
    y_pred = dnn.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
#    print(f'Scikit: {dnn.score(X_test, y_test_1hot)}')


#    neural = MultilayerPerceptron(hidden_layer_sizes=hl,
#            eta=0.001, 
#            learning_rate='constant',
#            alpha=0.0,
#            batch_size=100,
#            n_epochs=200,
#            act_func_str='sigmoid',
#            output_func_str='sigmoid',
#            cost_func_str='crossentropy')
#
#    neural.fit(X_train, y_train_1hot)
#    y_pred = neural.predict(X_test)
#    print(f'Our code: {accuracy_score(y_test, y_pred)}')









    #sklearn
    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X_train_val, y_train_val)
    # pred_skl = clf.predict(X_test)


    # pred_train = model.predict(X_train_val, probability=True)
    # pred_test = model.predict(X_test, probability=True)
    # area_ratio_train = cumulative_gain_area_ratio(y_train_val, pred_train, title='training results')
    # area_ratio_test = cumulative_gain_area_ratio(y_test, pred_test, title='test results')
    # print('area ratio train:', area_ratio_train)
    # print('area ratio test:', area_ratio_test)



    ax1 = plot_confusion_matrix(y_test, y_pred, normalize=True, cmap='Blues')
    # ax2 = plot_confusion_matrix(y_test, pred_skl, normalize=True, cmap='Reds')


    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom + 0.5, top - 0.5)
    # ax2.set_ylim(bottom + 0.5, top - 0.5)

    plt.show()



def nn_regression(X, y):
    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)


    hl = [100,30,20]
    neural = MultilayerPerceptron(hidden_layer_sizes=hl,
            eta=0.001, alpha=0.000, batch_size=100,
            learning_rate='constant',
            n_epochs=100, 
            act_func_str='sigmoid',
            cost_func_str='mse',
            output_func_str='identity')

    neural.fit(X_train, y_train)
    y_pred = neural.predict_probabilities(X)

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
##       batch_size=200, 
#        tol=1e-7,
#        learning_rate='constant')
#    dnn.fit(X_train, y_train)
#    y_pred = dnn.predict(X)

#    print(f'Scikit: {r2_score(y_test, y_pred)}')
#    print(f'Scikit MSE: {mean_squared_error(y_test, y_pred)}')

    n = 201
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
    X_c, y_c, scale_columns = preprocess_CC_data('../data/credit_card.xls')
#    regression_analysis(X_f, y_f)
#    logistic_analysis(X_b, y_b)
#    nn_classification(X_b, y_b)
    nn_classification(X_c, y_c, scale_columns)
#    nn_regression(X_f, y_f)
