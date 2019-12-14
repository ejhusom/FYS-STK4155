import numpy as np
import pandas as pd
from RegressionMethods import *
from Resampling import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from imageio import imread
import seaborn as sns


def load_terrain(filename):
    """
    Function to load terrain data and convert it to a 2D array
    Returns a rescaled version of the array and its dimensions.
    """
    slice = 2
    terrain = imread('data/' + filename)
    dims = np.shape(terrain)
    print(dims)
    if dims[0] != dims[1]:
        terrain = terrain[0:dims[1], :]
        dims = terrain.shape
    terrain = terrain[0:dims[0]//2, 0:dims[1]//2]
    terrain = terrain[0:-1:slice, 0:-1:slice]
    dims = np.shape(terrain)
    print(filename, 'loaded.', dims[0],'x',dims[1])
    return terrain*0.001, dims[0]


def show_terrain(terrain_data):
    """
    Simple function for showing the raw terrain data
    """
    terrain1 = imread(terrain_data)
    plt.figure()
    plt.title('Terrain')
    plt.imshow(terrain1, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def generate_mesh(n, random_pts = 0):
    """
    Generated a mesh of n x and y values.
    """
    if random_pts == 0:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
    if random_pts == 1:
        x = np.random.rand(n)
        y = np.random.rand(n)
    return np.meshgrid(x, y)


def frankie_function(x, y, n, sigma = 0, mu = 0):
    """
    Calculates the values of the Franke function.
    Returns these values with an element of noise if specified.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(mu, sigma, n)


def create_design_matrix(x, y, deg):
    """
    Creates a design matrix with columns:
    [1  x  y  x^2  y^2  xy  x^3  y^3  x^2y ...]
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((deg + 1)*(deg + 2)/2)
    X = np.ones((N,p))

    for i in range(1, deg + 1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X



def plot_model(x, y, z, model, deg):
    """
    Function for plotting a height map of the predicted output values of a model,
    and the raw terrain data.

    Inputs:
    -model, instanciated model
    -deg, degree of the model
    """

    #perform regression on the data and predict
    X = create_design_matrix(x, y, deg)
    z_flat = np.ravel(z)
    model.fit(X, z_flat)
    z_pred = model.predict(X).reshape(len(x), len(y))

    #plotting
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, projection='3d', gridspec_kw = {'wspace':0, 'hspace':0})
    ax2 = fig.add_subplot(1, 2, 2, projection='3d', gridspec_kw = {'wspace':0, 'hspace':0})
    surf1 = ax1.plot_surface(x, y, z_pred, cmap=cm.terrain, alpha=0.9)
    surf2 = ax2.plot_surface(x, y, z, cmap=cm.terrain, alpha=0.9)
    ax1.set_zlim(0.75, 1.75)
    ax2.set_zlim(0.75, 1.75)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Height [km]')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Height [km]')
    ax1.title.set_text('Modelled terrain')
    ax2.title.set_text('Actual terrain')
    plt.show()
