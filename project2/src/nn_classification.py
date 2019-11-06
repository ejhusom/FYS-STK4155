#!/usr/bin/env python3
# ============================================================================
# File:     nn_classification.py
# Author:   Erik Johannes Husom
# Created:  2019-11-06
# ----------------------------------------------------------------------------
# Description:
# Analyze performance of neural network applied to classification problems.
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import time

import sklearn as skl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from scikitplot.metrics import plot_confusion_matrix
import scikitplot.metrics as skplt

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

def logistic_analysis(X, y):

    print(CV(X, y, SGDClassifier(), n_splits=10))       # sklearn
    print(CV(X, y, SGDClassification(), n_splits=10))   # pylearn


def nn_classification_simple():

    scale_columns = None
    #X, y, scale_columns = preprocess_CC_data('../data/credit_card.xls')
    X, y = breast_cancer_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    X_train, X_test = scale_data(X_train, X_test, scaler='minmax')

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



    model = MultilayerPerceptron(
            hidden_layer_sizes=[10,10,10],
            eta=0.1, 
            learning_rate='constant',
            alpha=0.0,
            batch_size=100,
            n_epochs=200,
            weights_init='normal',
            act_func_str='sigmoid',
            output_func_str='sigmoid',
            cost_func_str='crossentropy')

    model.fit(X_train, y_train_1hot)
    y_pred = model.predict_class(X_test)
    print(f'pylearn accuracy: {accuracy_score(y_test, y_pred)}')
    nn_classification_plot(y_test, y_pred)


def nn_classification_analysis(train=False):

    #balance_outcomes = True

    #X, y, scale_columns = preprocess_CC_data('../data/credit_card.xls',
    #        which_onehot=1)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


    ##balance training set such that outcomes are 50/50
    #if balance_outcomes:
    #    non_default_inds = np.where(y_train==0)[0]
    #    default_inds = np.where(y_train==1)[0]

    #    remove_size = len(non_default_inds) - len(default_inds)
    #    remove_inds = np.random.choice(non_default_inds, size=remove_size, replace=False)


    #    X_train = np.delete(X, remove_inds, axis=0)
    #    y_train = np.delete(y, remove_inds, axis=0)



    #minmaxscaler = MinMaxScaler()
    #scaler = ColumnTransformer(
    #                    remainder='passthrough',
    #                    transformers=[('minmaxscaler', minmaxscaler, scale_columns)])



    #scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    #y_train = y_train.reshape(-1,1)
    #encoder = OneHotEncoder(categories='auto')
    #y_train_1hot = encoder.fit_transform(y_train).toarray()
    #y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

    X, X_train, X_test, y, y_train, y_train_1hot, y_test, y_test_1hot = \
        credit_card_train_test('../data/credit_card.xsl')

    # Test cases
    etas = np.logspace(-1, -4, 4)                 # 0.1, 0.01, ...
    n_epochs = [10, 100, 250, 500]             
    layers = [100,100,100]

    accuracy_eta = np.zeros(len(etas))

    if train:
        i = 0
        for eta in etas:

            model = MultilayerPerceptron(
                        hidden_layer_sizes=layers,
                        eta=eta, 
                        alpha=0.0, 
                        batch_size=100,
                        learning_rate='constant',
                        n_epochs=100, 
                        weights_init='normal',
                        act_func_str='sigmoid',
                        cost_func_str='crossentropy',
                        output_func_str='sigmoid')

            model.fit(X_train, y_train_1hot)
            y_pred_test = model.predict_class(X_test)
            accuracy_eta[i] = accuracy_score(y_test, y_pred_test)
            print(f'Accuracy: {accuracy_eta[i]}')
            i += 1
            print(f'Eta={eta} done')


        eta_opt = etas[np.argmax(accuracy_eta)]
        print(eta_opt)
        print(f'Optimal eta: {eta_opt}')

        timestr = time.strftime('%Y%m%d-%H%M%S')
        # np.save(timestr + '-accuracy_eta', accuracy_eta)
        np.save('class_accuracy_eta', accuracy_eta)


        accuracy_epoch = np.zeros(len(n_epochs))
        i = 0
        for n in n_epochs:

            model = MultilayerPerceptron(
                        hidden_layer_sizes=layers,
                        eta=eta_opt, 
                        alpha=0.0, 
                        batch_size=100,
                        learning_rate='constant',
                        n_epochs=n, 
                        weights_init='normal',
                        act_func_str='sigmoid',
                        cost_func_str='crossentropy',
                        output_func_str='sigmoid')

            model.fit(X_train, y_train_1hot)
            y_pred_test = model.predict_class(X_test)
            accuracy_epoch[i] = accuracy_score(y_test, y_pred_test)
            i += 1
            print(f'Epochs={n} done')

        np.save('class_accuracy_epoch', accuracy_epoch)


    accuracy_eta = np.load('class_accuracy_eta.npy')
    accuracy_epoch = np.load('class_accuracy_epoch.npy')
    

    fig = plt.figure(figsize=(9.5,4.5))

    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(r'$\log_{10}$ of Learning rate')
    ax1.set_ylabel('Accuracy score')
    ax1.plot(np.log10(etas), accuracy_eta, '.-', label='sigmoid')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('number of epochs')
    ax2.plot(n_epochs, accuracy_epoch, '.-', label='sigmoid')
    ax2.legend()

    plt.savefig('class-eta-accuracy.pdf')
    plt.show()
    




def nn_classification_heatmap(train=False):

    X, y, scale_columns = preprocess_creditcard_data('../data/credit_card.xls')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    

    # Test cases
    n_layers = np.arange(1, 10, 1)                  # 1, 2, 3, ...
    n_nodes = np.arange(10, 101, 10)                # 10, 20, 30, ...

    accuracy = np.zeros((len(n_layers), len(n_nodes)))

    if train:
        i = 0
        for l in n_layers:
            j = 0
            for n in n_nodes:
                layers = list(np.ones(l, dtype='int') * n)

                model = MultilayerPerceptron(
                            hidden_layer_sizes=layers,
                            eta=1e-1, 
                            alpha=0.0, 
                            batch_size=100,
                            learning_rate='constant',
                            n_epochs=500, 
                            act_func_str='relu',
                            cost_func_str='mse',
                            output_func_str='identity')

                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                if np.isnan(y_pred_test).any():
                    accuracy[i,j] = np.nan
                    print('Nan detected')
                else:
                    accuracy[i,j] = accuracy_score(y_test, y_pred_test)
                j += 1
                print(f'Nodes: {n}')
            i += 1
            print(f'Layers: {l}')

        np.save('accuracy_heat', accuracy)

    accuracy = np.load('accuracy_heat.npy')

    min_idcs = np.where(accuracy == np.nanmin(accuracy))
    print(min_idcs)

    plt.figure(figsize=(9.5,4.5))

    print(n_layers)
    print(n_nodes)
    ax = sns.heatmap(accuracy, annot=True, xticklabels=n_nodes, yticklabels=n_layers)
    ax.add_patch(Rectangle((min_idcs[1], min_idcs[0]), 1, 1, fill=False, edgecolor='red', lw=3))
    # ax.set_xticks(n_layers)
    ax.set_xlabel('Number of nodes per layer')
    ax.set_ylabel('Number of layers')
    # ax.set_yticks(n_nodes)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('heatmap.pdf')
    plt.show()


def nn_classification_optimal(train=False):
    X, X_train, X_test, y, y_train, y_train_1hot, y_test, y_test_1hot = \
        credit_card_train_test('../data/credit_card.xls')

    if train:
        i = 0
        model = MultilayerPerceptron(
                    hidden_layer_sizes=[100,100,100],
                    eta=1e-2, 
                    alpha=0.0, 
                    batch_size=100,
                    learning_rate='constant',
                    n_epochs=100, 
                    weights_init='normal',
                    act_func_str='sigmoid',
                    cost_func_str='crossentropy',
                    output_func_str='sigmoid')

        model.fit(X_train, y_train_1hot)
        y_pred_test = model.predict_class(X_test)
        np.save('class_y_pred_optimal', y_pred_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f'Accuracy: {accuracy}')




    y_pred = np.load('class_y_pred_optimal.npy')
    nn_regression_plot(X, y, y_pred)



def nn_classification_skl():
    X, y = breast_cancer_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    X_train, X_test = scale_data(X_train, X_test, scaler='minmax')
    
    # One hot encoding targets
    y_train = y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()
    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

    dnn = MLPClassifier(hidden_layer_sizes=[50,50,50], 
                        activation='logistic',
                        alpha=0.0, 
                        learning_rate_init=0.001, 
                        max_iter=1000,
                        batch_size=100, 
                        learning_rate='constant')
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




if __name__ == '__main__':
    pass

