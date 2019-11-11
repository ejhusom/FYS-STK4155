#!/usr/bin/env python3
# ============================================================================
# File:     nn_regression.py
# Created:  2019-10-22
# ----------------------------------------------------------------------------
# DESCRIPTION:
# Analyze performance of neural network applied to regression problems.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from pylearn.resampling import CV
from pylearn.linearmodel import Regression
from pylearn.multilayerperceptron import MultilayerPerceptron

from franke import *



def regression_analysis(X, y):
    """Performing Ridge regression, with cross-validation, using both
    Scikit-learn and pylearn.

    Parameters
    ----------
    X : array
        Design matrix.
    y : array
        Target vector.

    Returns
    -------
    Nothing.

    """

    print(CV(X, y, Ridge(alpha=0.00001), n_splits=20,
        classification=False))  # sklearn
    print(CV(X, y, Regression(method='ridge', alpha=0.00001), n_splits=20,
        classification=False))  # pylearn


def nn_regression_analysis(train=False):
    """Analyzing behaviour of the MSE of a model based on which
    learning parameter (eta) is used, and how many epochs we run the training.

    Parameters
    ----------
    train : boolean
        If True: Model is trained, and then used for results.
        If False: Model is assumed to be already trained, and stored arrays are
        used for producing results. The stored arrays needs to be present in
        the same directory is this function.

    Returns
    -------
    Nothing.

    """

    franke = FrankeDataset(n=20, eps=0.2)
    X, y = franke.generate_data_set()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    # Test cases
    etas = np.logspace(-1, -15, 15)                 # 0.1, 0.01, ...
    n_epochs = np.arange(0, 2001, 200)             
    n_epochs[0] = 1                                 # 1, 100, 200, ...
    act_funcs = ['relu', 'sigmoid']
    output_funcs = ['identity', 'identity']
    cost_funcs = ['mse', 'mse']
    layers = [100,100]

    eta_opt = np.zeros(2)

    mse_eta = np.zeros((len(etas), 2))
    r2_eta = np.zeros((len(etas), 2))

    if train:

        # k=0 -> relu 
        # k=1 -> sigmoid
        for k in range(2):
            i = 0
            for eta in etas:

                model = MultilayerPerceptron(
                            hidden_layer_sizes=layers,
                            eta=eta, 
                            alpha=0.0, 
                            batch_size=100,
                            learning_rate='constant',
                            n_epochs=500, 
                            act_func_str=act_funcs[k],
                            cost_func_str=cost_funcs[k],
                            output_func_str=output_funcs[k])

                # Skip 0.1 as eta for sigmoid, because of exploding gradient
                if (k == 1) and (eta == 0.1):
                    mse_eta[i,k] = np.nan
                    r2_eta[i,k] = np.nan
                else:
                    model.fit(X_train, y_train)
                    y_pred_test = model.predict(X_test)
                    mse_eta[i,k] = mean_squared_error(y_test, y_pred_test)
                    r2_eta[i,k] = r2_score(y_test, y_pred_test)
                i += 1
                print(f'Eta={eta}, k={k} done')

        eta_opt[0] = etas[np.argmin(mse_eta[:,0])]
        eta_opt[1] = etas[np.argmin(mse_eta[1:,1]) + 1]
        print(f'Optimal etas for relu/sigmoid: {eta_opt}')

        # Save MSE and R2 scores to uniquely named files
        timestr = time.strftime('%Y%m%d-%H%M%S')
        # np.save(timestr + '-mse_eta', mse_eta)
        # np.save(timestr + '-r2_eta', r2_eta)
        np.save('mse_eta', mse_eta)
        np.save('r2_eta', r2_eta)


        mse_epoch = np.zeros((len(n_epochs), 2))
        r2_epoch = np.zeros((len(n_epochs), 2))
        for k in range(2):
            i = 0
            for n in n_epochs:

                model = MultilayerPerceptron(
                            hidden_layer_sizes=layers,
                            eta=eta_opt[k], 
                            alpha=0.0, 
                            batch_size=100,
                            learning_rate='constant',
                            n_epochs=n, 
                            act_func_str=act_funcs[k],
                            cost_func_str=cost_funcs[k],
                            output_func_str=output_funcs[k])

                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                mse_epoch[i,k] = mean_squared_error(y_test, y_pred_test)
                r2_epoch[i,k] = r2_score(y_test, y_pred_test)
                i += 1
                print(f'Epochs={n}, k={k} done')

        np.save('mse_epoch', mse_epoch)
        np.save('r2_epoch', r2_epoch)


    mse_eta = np.load('mse_eta.npy')
    r2_eta = np.load('r2_eta.npy')
    mse_epoch = np.load('mse_epoch.npy')
    r2_epoch = np.load('r2_epoch.npy')
    

    fig = plt.figure(figsize=(9.5,4.5))

    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(r'$\log_{10}$ of Learning rate')
    ax1.set_ylabel('Mean Squared Error')
    ax1.plot(np.log10(etas), mse_eta[:,0], '.-', label='relu')
    ax1.plot(np.log10(etas), mse_eta[:,1], '.-', label='sigmoid')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('number of epochs')
    ax2.set_ylabel('Mean Squared Error')
    ax2.plot(n_epochs, mse_epoch[:,0], '.-', label='relu')
    ax2.plot(n_epochs, mse_epoch[:,1], '.-',label='sigmoid')
    ax2.legend()

    plt.savefig('eta-mse.pdf')


def nn_regression_heatmap(train=False):
    """Grid search for optimal hidden layer configuration in neural network.

    Parameters
    ----------
    train : boolean
        If True: Model is trained, and then used for results.
        If False: Model is assumed to be already trained, and stored arrays are
        used for producing results. The stored arrays needs to be present in
        the same directory is this function.

    Returns
    -------
    Nothing.

    """

    franke = FrankeDataset(n=20, eps=0.2)
    X, y = franke.generate_data_set()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    

    # Test cases
    n_layers = np.arange(1, 10, 1)                  # 1, 2, 3, ...
    n_nodes = np.arange(10, 101, 10)                # 10, 20, 30, ...

    mse = np.zeros((len(n_layers), len(n_nodes)))
    r2 = np.zeros((len(n_layers), len(n_nodes)))

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
                    mse[i,j] = np.nan
                    r2[i,j] = np.nan
                    print('Nan detected')
                else:
                    mse[i,j] = mean_squared_error(y_test, y_pred_test)
                    r2[i,j] = r2_score(y_test, y_pred_test)
                j += 1
                print(f'Nodes: {n}')
            i += 1
            print(f'Layers: {l}')

        np.save('mse_heat', mse)
        np.save('r2_heat', r2)

    mse = np.load('mse_heat.npy')
    r2 = np.load('r2_heat.npy')

    min_idcs = np.where(mse == np.nanmin(mse))
    print(min_idcs)

    plt.figure(figsize=(9.5,4.5))

    print(n_layers)
    print(n_nodes)
    ax = sns.heatmap(mse, annot=True, xticklabels=n_nodes, yticklabels=n_layers)
    ax.add_patch(Rectangle((min_idcs[1], min_idcs[0]), 1, 1, fill=False, edgecolor='red', lw=3))
    # ax.set_xticks(n_layers)
    ax.set_xlabel('Number of nodes per layer')
    ax.set_ylabel('Number of layers')
    # ax.set_yticks(n_nodes)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('heatmap.pdf')
    plt.show()


def nn_regression_optimal(train=False):
    """Training a model with given parameters, with a high number of epochs.

    Parameters
    ----------
    train : boolean
        If True: Model is trained, and then used for results.
        If False: Model is assumed to be already trained, and stored arrays are
        used for producing results. The stored arrays needs to be present in
        the same directory is this function.

    Returns
    -------
    Nothing.

    """

    franke = FrankeDataset(n=20, eps=0.2)
    X, y = franke.generate_data_set()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    if train:
        model = MultilayerPerceptron(
                    hidden_layer_sizes=[70,70,70],
                    eta=1e-1, 
                    alpha=0.0, 
                    batch_size=100,
                    learning_rate='constant',
                    n_epochs=2000, 
                    act_func_str='relu',
                    cost_func_str='mse',
                    output_func_str='identity')

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred = model.predict(X)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        print(f'MSE: {mse}')
        print(f'R2: {r2}')
        
        np.save('y_pred_optimal', y_pred)

    y_pred = np.load('y_pred_optimal.npy')
    nn_regression_plot(X, y, y_pred)
    


def nn_regression_gridsize(train=False):
    """Analyzing behaviour of MSE and R2 based on the size of the Franke data
    set, and how much noise we have introduced to the data.
    train : boolean
        If True: Model is trained, and then used for results.
        If False: Model is assumed to be already trained, and stored arrays are
        used for producing results. The stored arrays needs to be present in
        the same directory is this function.

    Returns
    -------
    Nothing.

    """

    gridsizes = [20,40,60,80]
    noises = [0.0, 0.1, 0.2]
    noises = [0.2]

    mses = np.zeros((len(noises), len(gridsizes)))
    mses_train = np.zeros((len(noises), len(gridsizes)))
    r2s = np.zeros((len(noises), len(gridsizes)))
    biases = np.zeros((len(noises), len(gridsizes)))
    variances = np.zeros((len(noises), len(gridsizes)))

    if train:
        i = 0
        for eps in noises:
            j = 0
            for n in gridsizes:
                franke = FrankeDataset(n=n, eps=eps)
                X, y = franke.generate_data_set()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                        random_state = 0)

                model = MultilayerPerceptron(
                            hidden_layer_sizes=[70,70,70],
                            eta=1e-2, 
                            alpha=0.0, 
                            batch_size=100,
                            learning_rate='constant',
                            n_epochs=200, 
                            act_func_str='relu',
                            cost_func_str='mse',
                            output_func_str='identity')

                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)
                y_pred = model.predict(X)
                if np.isnan(y_pred_test).any():
                    mses[i,j] = np.nan
                    mses_train[i,j] = np.nan
                    r2s[i,j] = np.nan
                    biases[i,j] = np.nan
                    variances[i,j] = np.nan
                else:
                    mses[i,j] = mean_squared_error(y_test, y_pred_test)
                    mses_train[i,j] = mean_squared_error(y_train, y_pred_train)
                    r2s[i,j] = r2_score(y_test, y_pred_test)
                    biases[i,j] = np.mean((y_test - np.mean(y_pred_test)) ** 2)
                    variances[i,j] = np.var(y_pred_test)
                print(f'Grid size: {n}')
                print(f'MSE: {mses[i,j]}')
                print(f'R2: {r2s[i,j]}')
                j += 1
            i += 1

            # np.save(f'y_pred_optimal_grid{n}', y_pred)
        
        np.save('mse_gridsize_analysis', np.array(mses))
        np.save('r2_gridsize_analysis', np.array(r2s))

    mses = np.load('mse_gridsize_analysis.npy')
    r2s = np.load('r2_gridsize_analysis.npy')
    
    fig = plt.figure(figsize=(9.5,4.5))

    ax1 = fig.add_subplot(121)
    i = 0
    for eps in noises:
        ax1.plot(gridsizes, mses[i,:], '.-', label=f'test; noise: {eps}')
        ax1.plot(gridsizes, mses_train[i,:], '.-', label=f'train; noise: {eps}')
        # ax1.plot(gridsizes, variances[i,:], '.-', label=f'variance; noise: {eps}')
        # ax1.plot(gridsizes, biases[i,:], '.-', label=f'bias; noise: {eps}')
        i += 1  
    ax1.set_xlabel('grid size')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    i = 0
    for eps in noises:
        ax2.plot(gridsizes, r2s[i,:], '.-', label=f'noise: {eps}')
        i += 1  
    ax2.set_xlabel('grid size')
    ax2.legend()

    plt.savefig('grid-mse.pdf')
    plt.show()



def nn_regression_skl():
    """Comparison of Scikit-Learn and pylearn."""

    franke = FrankeDataset(n=100, eps=0.1)
    X, y = franke.generate_data_set()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    hl = [70,70,70]
    eta = 0.01
    batch_size = 100
    n_epochs = 1000


    model = MultilayerPerceptron(
                hidden_layer_sizes=hl,
                eta=eta, 
                alpha=0.0, 
                batch_size=batch_size,
                learning_rate='constant',
                n_epochs=n_epochs, 
                act_func_str='relu',
                cost_func_str='mse',
                output_func_str='identity')

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred = model.predict(X)
    print(f'pylearn R2: {r2_score(y_test, y_pred_test)}')
    print(f'pylearn MSE: {mean_squared_error(y_test, y_pred_test)}')
    nn_regression_plot(X, y, y_pred)




    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    dnn = MLPRegressor(
            hidden_layer_sizes=hl, 
            alpha=0.0, 
            learning_rate_init=eta,
            max_iter=n_epochs,
            batch_size=batch_size, 
            learning_rate='constant')

    dnn.fit(X_train, y_train)
    y_pred = dnn.predict(X)
    y_pred_test = dnn.predict(X_test)
    print(f'sklearn R2: {r2_score(y_test, y_pred_test)}')
    print(f'sklearn MSE: {mean_squared_error(y_test, y_pred_test)}')
    nn_regression_plot(X, y, y_pred)



def nn_regression_plot(X, y, y_pred):
    """3D-ploto of Franke function together with the predicted result, the
    former as a surface, the latter as a wiregrid.
    """

    n = int(np.sqrt(np.size(y)))
    x1 = X[:,0].reshape(n,n)
    x2 = X[:,1].reshape(n,n)
    y_mesh = y.reshape(n,n)
    y_pred_mesh = y_pred.reshape(n,n)

    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, y_mesh, cmap=cm.coolwarm)
    ax.plot_wireframe(x1, x2, y_pred_mesh)
    plt.savefig('nn_franke.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, np.abs(y_mesh - y_pred_mesh))
    plt.show()



if __name__ == '__main__':
    pass
