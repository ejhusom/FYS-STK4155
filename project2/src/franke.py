#!/usr/bin/env python3
# ============================================================================
# File:     franke.py
# Author:   Erik Johannes Husom
# Created:  2019-09-18
# ----------------------------------------------------------------------------
# Description:
# Generate data with the Franke function.
# ============================================================================
import numpy as np

def generate_mesh(start=0, stop=1, n=100):
    '''Generate x and y data and return at as a flat meshgrid.'''

    x1 = np.linspace(start, stop, n)
    x2 = np.linspace(start, stop, n)
    x1, x2 = np.meshgrid(x1, x2)
    
    return x1, x2


def franke_function(x1, x2, eps = 0.05):

    np.random.seed(0)

    n = len(x1)

    term1 = 0.75*np.exp(-(0.25*(9*x1-2)**2) - 0.25*((9*x2-2)**2))
    term2 = 0.75*np.exp(-((9*x1+1)**2)/49.0 - 0.1*(9*x2+1))
    term3 = 0.5*np.exp(-(9*x1-7)**2/4.0 - 0.25*((9*x2-3)**2))
    term4 = -0.2*np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    
    y = term1 + term2 + term3 + term4 + eps*np.random.randn(n)

    return y


def create_design_matrix(x1, x2, deg=5):

    p = int((deg+1)*(deg+2)/2)
    if len(x1.shape) > 1:
        x1 = np.ravel(x1)
        x2 = np.ravel(x2)

    N = len(x1)
    X = np.ones((N,p))

    for i in range(1, deg+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x1**(i-k) * x2**k

    return X

def plot(x1, x2, y):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x1, x2, y,
            cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
