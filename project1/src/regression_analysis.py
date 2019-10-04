#!/usr/bin/env python3
# ============================================================================
# File:     regression_analysis.py
# Author:   Erik Johannes Husom
# Created:  2019-09-11
# ----------------------------------------------------------------------------
# Description:
# Analyzing regression methods.
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from imageio import imread
import pandas as pd

from Regression import *
from crossvalidation import *
from franke import *
from designmatrix import *


def analyze_regression(x1, x2, y, method='ols', n_folds=5, data_name='data'):

    max_degree = 20
    n_lambdas = 9
    lambdas = np.logspace(-3, 3, n_lambdas)

    error_scores = pd.DataFrame(columns=['degree', 'lambda', 'MSE_train',
        'MSE_test', 'R2_train', 'R2_test', 'bias_train', 'bias_test',
        'var_train', 'var_test'])

    if method=='ols':
        lambdas = [0]
    
    filename = 'error_scores_' + data_name + '_' + method


    if n_folds > 1: 
        filename += '_cv'

    for lambda_ in lambdas:
        for deg in range(1, max_degree+1):
            #print(deg)
            X = create_design_matrix(x1, x2, deg=deg)

            if n_folds > 1:
                mse_train, mse_test, r2_train, r2_test, bias_train, bias_test, var_train, var_test = cross_validation(X, y, n_folds, method, lambda_)
                
                
            else:
                model = Regression(method, lambda_=lambda_)
                model.fit(X, y)
                model.predict(X)
                mse_train = mean_squared_error(model.y, model.y_pred)
                r2_train = r2_score(model.y, model.y_pred)
                bias_train = bias(model.y, model.y_pred)
                var_train = np.var(model.y_pred)
                mse_test = None
                r2_test = None
                bias_test = None
                var_test = None



            error_scores = error_scores.append({'degree': deg, 
                                                'lambda': lambda_, 
                                                'MSE_train': mse_train, 
                                                'MSE_test': mse_test,
                                                'R2_train': r2_train, 
                                                'R2_test': r2_test,
                                                'bias_train': bias_train,
                                                'bias_test': bias_test,
                                                'var_train': var_train,
                                                'var_test': var_test},
                                                ignore_index=True)

    

    print(error_scores)
    error_scores.to_csv(filename + '.csv')


def franke_regression():
    x1, x2 = generate_mesh(0, 1, 20)
    y = franke_function(x1, x2, eps=4.0)

    analyze_regression(x1, x2, y, n_folds=5, method='lasso', data_name='franke')
    

def terrain_regression(terrain_file, plot=0):

    terrain = imread(terrain_file)

    if plot==1:
        plt.figure()
        plt.imshow(terrain, cmap='gray')
        plt.show()


def plot_regression_analysis(filename):
    df = pd.read_csv(filename)


    lambdas = df['lambda'].unique()


    plt.figure()

    for lambda_ in lambdas:
        dfl = df.loc[df['lambda'] == lambda_]
        #plt.plot(dfl['degree'], dfl['MSE_train'], label=f'train, \
        #        lambda={lambda_}')
        plt.plot(dfl['degree'], dfl['MSE_test'], label=f'test, \
                lambda={lambda_}')


    # Plot bias/variance/error
    #plt.plot(df['degree'], df['bias_test'], label='bias')
    #plt.plot(df['degree'], df['var_test'], label='variance')
    #plt.plot(df['degree'], df['MSE_test'], label='error')

    # Plot MSE vs model complexity
    #plt.plot(df['degree'], df['MSE_test'], label='test')
    #plt.plot(df['degree'], df['MSE_train'], label='train')
    


    plt.legend()

    plt.show()
   



if __name__ == '__main__': 

    #terrain_regression('dat/n27_e086_1arc_v3.tif', plot=1)
    franke_regression()
    plot_regression_analysis('error_scores_franke_lasso _cv.csv')
