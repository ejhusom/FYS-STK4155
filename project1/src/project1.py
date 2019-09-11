#!/usr/bin/env python3
# ============================================================================
# File:     project1.py
# Author:   Erik Johannes Husom
# Created:  2019-09-11
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression



class RegressionResampling():

    def __init__(self, n=100, deg=5):

        self.n = n
        self.deg = deg
        self.x, self.y = self.generate_xy()
        self.z = self.franke_function()
        self.X = self.create_design_matrix()

        self.beta_OLS = self.OLS()
        self.z_OLS = self.X @ self.beta_OLS

        print(np.shape(self.z))
        print(np.shape(self.z_OLS))
        self.print_error_analysis(np.ravel(self.z), self.z_OLS)

    def franke_function(self, eps = 0.05):

        np.random.seed(0)

        x = self.x
        y = self.y

        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        
        return term1 + term2 + term3 + term4 + eps*np.random.randn(self.n)


    def generate_xy(self, start = 0, stop = 1):

        x = np.linspace(start, stop, self.n)
        y = np.linspace(start, stop, self.n)
        x, y = np.meshgrid(x, y)

        return x, y

    def create_design_matrix(self):

        if len(self.x.shape) > 1:
                x_rvl = np.ravel(self.x)
                y_rvl = np.ravel(self.y)

        N = len(x_rvl)
        p = int((self.deg+1)*(self.deg+2)/2)
        X = np.ones((N,p))

        for i in range(1, self.deg+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = x_rvl**(i-k) * y_rvl**k

        return X


    def OLS(self):

        z_rvl = np.ravel(self.z)
        X = self.X
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z_rvl)
        
        return beta


    def print_error_analysis(self, z, z_tilde):
        '''Print error analysis of regression fit using scikit.'''
        
        print("Mean squared error: %.2f" % mean_squared_error(z, z_tilde))
        print('R2 score: %.2f' % r2_score(z, z_tilde))
        print('Mean absolute error: %.2f' % mean_absolute_error(z, z_tilde))


    def cross_validation(self):

        k = 5

        for


    def plot_franke(self):

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(self.x, self.y, self.z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


if __name__ == '__main__': 
    proj = RegressionResampling()

