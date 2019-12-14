from RegressionMethods import *
from Resampling import *
from functions import *
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from matplotlib.patches import Rectangle
import time
plt.style.use('ggplot')





def model_degree_analysis(x, y, z, model_name, min_deg=1, max_deg=10, n_bootstraps = 100, alpha = 0, ID = '000'):
    """
    Function for analyzing the performance of a model for different model complexities.
    Performs bootstrap resampling for each configuration.
    Plots the MSE of the training error and test error for each configuration,
    and the bias^2 - variance decomposition of the testing error.
    The error scores for each configuration is also saved to a .csv file.


    Inputs:
    -x, y values, dimensions (n, n)
    -z values, dimension (n^2, 1)
    -model_name, name of the model: 'ols', 'ridge', 'lasso'
    -min_deg, max_deg - degrees to analyze
    -n_bootstraps - number of resamples in bootstrap
    -alpha - hyperparameter in 'ridge' and 'lasso'
    -ID - figure IDs
    """


    #setup directories
    dat_filename = 'results/' + 'error_scores_deg_analysis_' + model_name
    fig_filename = 'figures/' + 'deg_analysis_' + model_name
    error_scores = pd.DataFrame(columns=['degree', 'mse', 'bias', 'variance', 'r2', 'mse_train'])

    #initialize regression model and arrays for saving error values
    model = RegressionMethods(model_name, alpha=alpha)
    degrees = np.linspace(min_deg, max_deg, max_deg - min_deg + 1)
    nDegs = len(degrees)
    mse = np.zeros(nDegs)
    bias = np.zeros(nDegs)
    variance = np.zeros(nDegs)
    r2 = np.zeros(nDegs)
    mse_train = np.zeros(nDegs)


    min_mse = 1e100
    min_r2 = 0
    min_deg = 0
    i = 0

    #loop through the specified degrees to be analyzed
    for deg in degrees:
        X = create_design_matrix(x, y, int(deg))
        resample = Resampling(X, z)

        #perform bootstrap resampling and save error values
        mse[i], bias[i], variance[i], mse_train[i], r2[i] = resample.bootstrap(model, n_bootstraps)

        #save to pandas dataframe
        error_scores = error_scores.append({'degree': degrees[i],
                                            'mse': mse[i],
                                            'bias': bias[i],
                                            'variance': variance[i],
                                            'r2': r2[i],
                                            'mse_train': mse_train[i]}, ignore_index=True)

        #check if this configuration gives smallest error
        if mse[i] < min_mse:
            min_mse = mse[i]
            min_r2 = r2[i]
            min_deg = deg


        i += 1
    #end for


    #plot error of test set and training set
    plt.plot(degrees, mse, label='test set')
    plt.plot(degrees, mse_train, label='training set')
    plt.legend()
    plt.xlabel('Model complexity [degree]')
    plt.ylabel('Mean Squared Error')
    plt.savefig(fig_filename + '_test_train_' + ID + '.pdf')
    plt.show()

    #plot bias^2 variance decomposition of the test error
    plt.plot(degrees, mse, label='mse')
    plt.plot(degrees, bias,'--', label='bias')
    plt.plot(degrees, variance, label='variance')
    plt.xlabel('Model complexity [degree]')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(fig_filename + '_bias_variance_' + ID + '.pdf')
    plt.show()


    #save error scores to file
    error_scores.to_csv(dat_filename + '.csv')
    print('min mse:', min_mse)
    print('r2:', min_r2)
    print('deg:', min_deg)



def ridge_lasso_complexity_analysis(x, y, z, model_name, min_deg=1, max_deg=10, n_lambdas=13, min_lamb=-10, max_lamb=2, ID = '000'):
    """
    Function for analyzing ridge or lasso model performance for
    different values lambda and model complexity.
    Performs bootstrap resampling for each configuration and plots
    a heat map of the error for each configuration. Error scores are
    also saved to a .csv file.

    Inputs:
    -x, y values, dimensions (n, n)
    -z values, dimension (n^2, 1)
    -model_name, name of the model: 'ridge', 'lasso'
    -min_deg, max_deg - degrees to analyze
    -min_lamb, max_lamb, n_lambdas - values of log_10(lambda) to analyze, and how many
    -ID - figure IDs
    """


    #initialize model and arrays for parameters lambda and complexity
    model = RegressionMethods(model_name)
    lambdas = np.linspace(min_lamb, max_lamb, n_lambdas)
    degrees = np.linspace(min_deg, max_deg, max_deg - min_deg + 1)

    #setup directories
    dat_filename = 'results/' + 'error_scores_' + model_name
    fig_filename = 'figures/' + 'min_mse_meatmap_' + model_name
    error_scores = pd.DataFrame(columns=['degree', 'log lambda', 'mse', 'bias', 'variance', 'r2', 'mse_train'])


    min_mse = 1e100
    min_lambda = 0
    min_degree = 0
    min_r2 = 0


    i = 0
    #loop through specified degrees
    for deg in degrees:
        j = 0
        X = create_design_matrix(x, y, int(deg))
        resample = Resampling(X, z)
        #loop through specified lambdas
        for lamb in tqdm(lambdas):
            model.set_alpha(10**lamb)

            #perform resampling
            mse, bias, variance, mse_train, r2 = resample.bootstrap(model, n_bootstraps=10)

            #save error scores in pandas dataframe
            error_scores = error_scores.append({'degree': deg,
                                                'log lambda': lamb,
                                                'mse': mse,
                                                'bias': bias,
                                                'variance': variance,
                                                'r2': r2,
                                                'mse_train': mse_train}, ignore_index=True)

            #check if current configuration gives minimal error
            if mse < min_mse:
                min_mse = mse
                min_lambda = lamb
                min_degree = deg
                min_r2 = r2

            j+=1
        #end for lambdas
        i+=1
    #end for degrees

    print('min mse:', min_mse)
    print('min r2:', min_r2)
    print('degree:', min_degree)
    print('lambda:', min_lambda)


    #save scores to file
    error_scores.to_csv(dat_filename + '.csv')



    #plot heat map of error scores of each configuration
    mse_table = pd.pivot_table(error_scores, values='mse', index=['degree'], columns='log lambda')
    idx_i = np.where(mse_table == min_mse)[0]
    idx_j = np.where(mse_table == min_mse)[1]

    fig = plt.figure()
    ax = sns.heatmap(mse_table, annot=True, fmt='.2g', cbar=True, linewidths=1, linecolor='white',
                            cbar_kws={'label': 'Mean Squared Error'})
    ax.add_patch(Rectangle((idx_j, idx_i), 1, 1, fill=False, edgecolor='red', lw=3))

    ax.set_xlabel(r"$\log_{10}(\lambda)$")
    ax.set_ylabel("Complexity")
    ax.set_ylim(len(degrees), 0)
    plt.show()





def confidence_intervals(x, y, z_flat, model, degree, alpha = 0, noise = 0):
    """
    Function for finding the estimated confidence intervals of a given models beta-parameters,
    and makes a plot of the parameters with confidence intervals corresponing to
    a 95% confidence interval.
    """
    X = create_design_matrix(x, y, degree)
    resample = Resampling(X, z_flat)
    betas, variance = resample.bootstrap(model, get_beta_var=True)

    CI = 1.96*np.sqrt(variance)


    #plotting
    plt.xticks(np.arange(0, len(betas), step=1))
    plt.errorbar(range(len(betas)), betas, CI, fmt="b.", capsize=3, label=r'$\beta_j \pm 1.96 \sigma$')
    plt.legend()
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.grid()
    plt.show()



def confidence_interval_ols(X, z):
    """
    Function for finding the confidence intervals of the OLS beta-parameters.
    Uses the analytical expression of the variance. Makes a plot of the beta-parameters with
    error bars corresponing to 95% confidence interval.
    """

    model = RegressionMethods('ols')
    model.fit(X, z)
    betas = model.beta
    cov = np.var(z)*np.linalg.pinv(X.T.dot(X))
    std_betas = np.sqrt(np.diag(cov))
    CI = 1.96*std_betas


    #plot results
    plt.errorbar(range(len(betas)), betas, CI, fmt="b.", capsize=3, label=r'$\beta_j \pm 1.96 \sigma$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.grid()
    plt.show()
