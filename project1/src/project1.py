#!/usr/bin/env python3
# ============================================================================
# File:     project1.py
# Author:   Erik Johannes Husom
# Created:  2019-09-11
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
from Regression import *

def franke_function(eps = 0.05):

    np.random.seed(0)

    x = np.reshape(self.x, (self.n, self.n))
    y = np.reshape(self.y, (self.n, self.n))

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    z = term1 + term2 + term3 + term4 + eps*np.random.randn(self.n)

    return np.ravel(z)

def plot_franke(self):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(self.x, self.y, self.z,
            cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def ex_a(model):

    model.regression(model.X, model.z, lambda_=0)
    model.print_error_analysis(model.z, z_tilde)


def ex_b(model):

    model.cross_validation()


def ex_d(model):

    
    model.lasso(model.X, model.z, lambda_=0.1)
    model.print_error_analysis(model.z, z_tilde)


if __name__ == '__main__': 
    project1 = Regression()

    ex_a(project1)
    ex_b(project1)
    ex_d(project1)

