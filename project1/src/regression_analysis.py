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


def generate_xy(start=0, stop=1, n=100):
    '''Generate x and y data and return at as a flat meshgrid.'''

    x = np.linspace(start, stop, n)
    y = np.linspace(start, stop, n)
    x, y = np.meshgrid(x, y)
    
    return x, y


def franke_function(x, y, eps = 0.05):

    np.random.seed(0)

    n = len(x)

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    z = term1 + term2 + term3 + term4 + eps*np.random.randn(n)

    return z


def plot_franke(x, y, z):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, z,
            cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def create_design_matrix(x, y, deg=5):

    n = len(x)
    p = int((deg+1)*(deg+2)/2)
    x = np.ravel(x)
    y = np.ravel(y)

    N = len(x)
    X = np.ones((N,p))

    for i in range(1, deg+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X


def create_design_matrix2(x, y, deg=5):

    n = len(x)
    p = int((deg+1)*(deg+2)/2) - 1
    x = np.ravel(x)
    y = np.ravel(y)

    N = len(x)
    X = np.ones((N,p))

    for i in range(1, deg+1):
        q = int((i)*(i+1)/2) - 1
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k
            

    return X



def analyze(x, y, z, method='ols'):

    max_degree = 5

    error_scores = pd.DataFrame(columns=['degree', 'MSE', 'R2'])

    for deg in range(1, max_degree+1):
        X = create_design_matrix(x, y, deg=deg)
        model = Regression(method)
        model.set_lambda(0.1)
        model.fit(X, z)
        model.predict(X)

        error_scores = error_scores.append({'degree': deg, 
                                            'MSE': model.get_mse(), 
                                            'R2': model.get_r2()},
                                            ignore_index=True)

    
    print(f'Analyzing {method}:')
    print(error_scores)
    error_scores.to_csv(f'error_scores_ols.csv')






def analyze_regression(x, y, z, method='ols', n_folds=5, data_name='data'):

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


def franke_regression():

    x, y = generate_xy(0, 1, 5)
    z = franke_function(x, y, 0.05)

    #analyze_regression(x, y, z, 'ols', data_name='franke')
    analyze(x, y, z, method='ols')
    


if __name__ == '__main__': 

    #terrain_regression('dat/n27_e086_1arc_v3.tif', plot=1)
    franke_regression()
