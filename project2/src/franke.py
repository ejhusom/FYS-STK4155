#!/usr/bin/env python3
# ============================================================================
# File:     franke.py
# Author:   Erik Johannes Husom
# Created:  2019-10-10
# Version:  2.0
# ----------------------------------------------------------------------------
# Description:
# Generate data with the Franke function.
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class FrankeDataset():
    """Generate dataset of Franke function.

    Parameters
    ----------
    mesh_start : float
    mesh_stop : float
    n : int
    eps : float


    Attributes
    ----------
    x1 : array
    x2 : array
    X : 2D array
    y : array

    """


    def __init__(self, mesh_start=0, mesh_stop=1, n=101, eps=0.01):
        
        self.mesh_start = mesh_start
        self.mesh_stop = mesh_stop
        self.n = n
        self.eps = eps

        self.generate_mesh()
        self.generate_y()
        self.generate_data_set()


    def generate_mesh(self):
        '''Generate x and y data and return at as a flat meshgrid.'''

        x1 = np.linspace(self.mesh_start, self.mesh_stop, self.n)
        x2 = np.linspace(self.mesh_start, self.mesh_stop, self.n)
        self.x1, self.x2 = np.meshgrid(x1, x2)


    def generate_y(self):

        x1 = self.x1
        x2 = self.x2

        term1 = 0.75*np.exp(-(0.25*(9*x1-2)**2) - 0.25*((9*x2-2)**2))
        term2 = 0.75*np.exp(-((9*x1+1)**2)/49.0 - 0.1*(9*x2+1))
        term3 = 0.5*np.exp(-(9*x1-7)**2/4.0 - 0.25*((9*x2-3)**2))
        term4 = -0.2*np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
        
        self.y_mesh = term1 + term2 + term3 + term4 + self.eps*np.random.randn(self.n)


    def generate_data_set(self):
        """Convert the x1/x2-meshgrid and y to a design matrix and a target
        vector.
        """

        self.X = np.c_[self.x1.ravel()[:, np.newaxis], self.x2.ravel()[:, np.newaxis]]
        self.y = self.y_mesh.ravel()[:, np.newaxis]

        return self.X, self.y


    def create_polynomial_design_matrix(self, deg=5):
        """Create a specialized design matrix for performing linear regression
        with polynomials of a given degree.

        Parameters
        ----------
        deg : int, default=5
            Polynomial degree.

        """

        p = int((deg+1)*(deg+2)/2)
        if len(self.x1.shape) > 1:
            x1 = np.ravel(self.x1)
            x2 = np.ravel(self.x2)

        X = np.ones((self.n,p))

        for i in range(1, deg+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = x1**(i-k) * x2**k

        self.X = X


    def plot_franke(self):
        """Plot the Franke function."""

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.x1, self.x2, self.y_mesh)
        plt.show()


if __name__ == '__main__':
    pass
