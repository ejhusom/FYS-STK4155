#!/usr/bin/env python3
# ============================================================================
# File:     regression_analysis.py
# Author:   Erik Johannes Husom
# Created:  2019-09-11
# ----------------------------------------------------------------------------
# Description:
# Analyzing regression methods.
# ============================================================================
from Regression import *
from Resampling import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from imageio import imread
import pandas as pd

from franke import *
from designmatrix import *




def franke_regression(method='ols'):

    max_degree = 5

    x, y = generate_xy(0, 1, 100)
    z = franke_function(x, y, 0.00)

    error_scores = pd.DataFrame(columns=['degree', 'MSE', 'R2'])

    for deg in range(1, max_degree+1):

        X = create_design_matrix(x, y, deg=deg)
        model = Regression(method)
        model.set_lambda(0.01)
#        model.fit(X, z)
#        model.predict(X)
        model.skl_fit(X, z)
        model.skl_predict(X)

        beta = model.beta
        y_predict = model.z_predict
        print(model.get_r2())

        print(beta)
        print(y_predict)




        error_scores = error_scores.append({'degree': deg, 
                                            'MSE': model.get_mse(), 
                                            'R2': model.get_r2()},
                                            ignore_index=True)

    
    print(f'Analyzing {method}:')
    print(error_scores)
    error_scores.to_csv(f'error_scores_{method}.csv')






def analyze_regression_cv(x, y, z, method='ols', n_folds=5, data_name='data'):

    max_degree = 5
    n_lambdas = 6
    lambdas = np.logspace(-3, 0, n_lambdas)

    error_scores = pd.DataFrame(columns=['degree', 'lambda', 'MSE_train',
        'MSE_test', 'R2_train', 'R2_test'])

    if method=='ols':
        lambdas = [0]


    for lambda_ in lambdas:
        for deg in range(1, max_degree+1):
            #print(deg)
            X = create_design_matrix(x, y, deg=deg)
            model = Resampling(method)

            model.cross_validation(X, z, n_folds, lambda_)

            error_scores = error_scores.append({'degree': deg, 
                                                'lambda': lambda_, 
                                                'MSE_train': model.mse_train, 
                                                'MSE_test': model.mse_test,
                                                'R2_train': model.r2_train, 
                                                'R2_test': model.r2_test},
                                                ignore_index=True)

    

    print(error_scores)
    error_scores.to_csv(f'error_scores_{data_name}_{method}_cv.csv')


def terrain_regression(terrain_file, plot=0):

    terrain = imread(terrain_file)

    if plot==1:
        plt.figure()
        plt.imshow(terrain, cmap='gray')
        plt.show()


if __name__ == '__main__': 

    #terrain_regression('dat/n27_e086_1arc_v3.tif', plot=1)
    franke_regression()
