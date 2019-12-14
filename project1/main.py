from RegressionMethods import *
from Resampling import *
from functions import *
from analysis import *
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.colors as colors
from imageio import imread



def main():
    """
    Main function for performing the analysis of the regression methods
    """
    np.random.seed(100)
    n = 20
    deg = 5
    sigma = 0.2

    ### frankie data
    # x, y = generate_mesh(n)
    # z = frankie_function(x, y, n, sigma)
    # z_flat = np.ravel(z)
    ###

    ### terrain data
    # terrain_data, n = load_terrain('norway1.tif')
    # z_flat = np.ravel(terrain_data)
    # x, y = generate_mesh(n)
    ###


    # plot_model(x, y, terrain_data, model, deg=10)
    # model_degree_analysis(x, y, z_flat, 'ols', min_deg=1, max_deg = 10, alpha = 10**-10)
    # ridge_lasso_complexity_analysis(x, y, z_flat, 'lasso',min_deg=1, max_deg=10)

if __name__ == '__main__':
    main()
