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

def plot(x, y, z):

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
