#!/usr/bin/env python3
# ============================================================================
# File:     nn_regression.py
# Author:   Erik Johannes Husom
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

    print(CV(X, y, Ridge(alpha=0.00001), n_splits=20,
        classification=False))  # sklearn
    print(CV(X, y, Regression(method='ridge', alpha=0.00001), n_splits=20,
        classification=False))  # pylearn


def nn_regression_analysis(train=False):

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
    plt.show()





def nn_regression_heatmap(train=False):

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
    franke = FrankeDataset(n=20, eps=0.2)
    X, y = franke.generate_data_set()
    franke_clean = FrankeDataset(n=200, eps=0.0)
    X_clean, y_clean = franke_clean.generate_data_set()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    


    if train:
        model = MultilayerPerceptron(
                    hidden_layer_sizes=[70,70,70],
                    eta=1e-2, 
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

    gridsizes = [20,50,100,200,500,1000]

    mses = []
    r2s = []

    if train:
        for n in gridsizes:
            franke = FrankeDataset(n=n, eps=0.2)
            X, y = franke.generate_data_set()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                    random_state = 0)

            model = MultilayerPerceptron(
                        hidden_layer_sizes=[50,50,50],
                        eta=0.01, 
                        alpha=0.0, 
                        batch_size=100,
                        learning_rate='constant',
                        n_epochs=500, 
                        act_func_str='relu',
                        cost_func_str='mse',
                        output_func_str='identity')

            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred = model.predict(X)
            if np.isnan(y_pred_test).any():
                mse = np.nan
                r2 = np.nan
            else:
                mse = mean_squared_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
            print(f'Grid size: {n}')
            print(f'MSE: {mse}')
            print(f'R2: {r2}')
            mses.append(mse)
            r2s.append(r2)

            # np.save(f'y_pred_optimal_grid{n}', y_pred)
        
        np.save('mse_gridsize_analysis', np.array(mses))
        np.save('r2_gridsize_analysis', np.array(r2s))

    mses = np.load('mse_gridsize_analysis.npy')
    r2s = np.load('r2_gridsize_analysis.npy')
    
    plt.figure()
    plt.plot(gridsizes, mses, '.-',label='sigmoid')
    plt.xlabel('grid size')
    plt.ylabel('Mean Squared Error')
    plt.savefig('eta-mse.pdf')
    plt.show()

        # y_pred = np.load('y_pred_optimal.npy')
        # nn_regression_plot(X, y, y_pred)



def nn_regression_skl(X, y):
    franke = FrankeDataset(n=20, eps=0.2)
    X, y = franke.generate_data_set()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)
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
