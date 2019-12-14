import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time


class Resampling:
    """
    Class for performing a train/test split and bootstrap resampling.
    Is initialized with the whole data set, splitting in training and testing
    data is performed by the methods.
    """
    def __init__(self, X, z):
        self.X = X.astype('float64')
        self.z = z.astype('float64')


    def train_test(self, model, test_size = 0.2):
        """
        Performs a simple train/test split and trains a model on the train data
        and returns the errors of the predictions on the test data.

        Inputs:
        -model, instanciated model
        -test_size, how much of the data to be used in testing
        """


        #split the data in training and test
        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size = test_size)

        #fit the model on the train data
        model.fit(X_train, z_train)

        #predict on the test data
        z_pred = model.predict(X_test)

        #calculate errors
        error = np.mean((z_test - z_pred)**2)
        bias = np.mean((z_test - np.mean(z_pred))**2)
        variance = np.var(z_pred)
        r2 = r2_score(z_test, z_pred)

        return error, bias, variance, r2


    def bootstrap(self, model, n_bootstraps = 100, test_size = 0.2, get_beta_var = False):
        """
        Performs bootstrap resampling and returns the error scores of both training and
        test data. Can also return the beta-parameters with its variance if specified.

        Inputs:
        -model, instanciated model
        -n_bootstraps, how many bootstrap resamplings to perform
        -test_size, how much of the data to be used in testing
        -get_beta_var, whether to return only beta values and their variance
        """

        #split data in training and testing
        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size = test_size)
        sampleSize = X_train.shape[0]
        n_betas = np.shape(self.X)[1]

        #setup arrays for storing prediction values
        z_pred = np.empty((z_test.shape[0], n_bootstraps))
        z_train_pred = np.empty((z_train.shape[0], n_bootstraps))
        z_train_boot = np.empty((z_train.shape[0], n_bootstraps))
        betas = np.empty((n_betas, n_bootstraps))
        r2 = np.empty(n_bootstraps)


        #perform the resamplings
        for i in range(n_bootstraps):

            #pick random values in the training data and fit the model to it
            indices = np.random.randint(0, sampleSize, sampleSize)
            X_, z_ = X_train[indices], z_train[indices]
            model.fit(X_, z_)

            #save z-values of the training set
            z_train_boot[:,i] = z_

            #predict on the same test data each time
            z_pred[:,i] = model.predict(X_test)
            z_train_pred[:,i] = model.predict(X_)
            betas[:,i] = model.beta
            r2[i] = r2_score(z_pred[:,i], z_test)


        z_test = z_test.reshape((len(z_test), 1))

        #calculate mean error scores
        error = np.mean( np.mean((z_pred - z_test)**2, axis=1, keepdims=True))
        error_train = np.mean( np.mean((z_train_pred - z_train_boot)**2, axis=1, keepdims=True))
        bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )


        beta_variance = np.var(betas, axis=1)
        betas = np.mean(betas, axis=1)


        if get_beta_var:
            return betas, beta_variance
        else:
            return error, bias, variance, error_train, np.mean(r2)
