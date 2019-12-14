import numpy as np
import sklearn.linear_model as skl



class RegressionMethods:
    """
    Class implementation of the OLS, Ridge and Lasso regression methods.
    Initiate with a specification of which method to use, and hyperparameter alpha for Ridge and Lasso.
    The OLS and Ridge methods find the analytical beta-parameters, the Lasso method is a call to
    sklearns Lasso method.
    """

    def __init__(self, method = 'ols', alpha = 0):
        self.method = method
        self.alpha = alpha


    def ols(self):
        """
        Finds the beta-parameters by OLS regression.
        In case of singular matrix, uses the pseudo inverse in
        calculation of beta.
        """
        XT = self.X.T
        self.beta = np.linalg.pinv(XT.dot(self.X)).dot(XT).dot(self.z)


    def ridge(self):
        """
        Finds the beta-parameters by Ridge regression.
        In case of singular matrix, uses the pseudo inverse in
        calculation of beta.
        """
        XT = self.X.T
        p = np.shape(self.X)[1]
        L = np.identity(p)*self.alpha
        self.beta = np.linalg.pinv(XT.dot(self.X) + L).dot(XT).dot(self.z)


    def lasso(self):
        clf = skl.Lasso(alpha = self.alpha, fit_intercept=False, normalize=False, max_iter=10000, tol=0.006).fit(self.X, self.z)
        self.beta = clf.coef_


    def fit(self, X, z):
        """
        Fits the specified model to the data. Makes a call to the
        relevant regression method.

        Inputs:
        -Design matrix X, dimension (n, p)
        -Target values z, dimension (n, 1)
        """
        self.X = X
        self.z = z
        if self.method == 'ols':
            self.ols()
        elif self.method == 'ridge':
            self.ridge()
        elif self.method == 'lasso':
            self.lasso()


    def predict(self, X):
        """
        Does a prediction on a set of data with the
        parameters beta found by the fit-method.

        Input:
        -Data to be predicted on, in a matrix (n, p)
        """
        self.z_tilde = X @ self.beta
        return self.z_tilde


    def set_alpha(self, alpha):
        """
        Change the alpha parameter after initialiation.
        """
        self.alpha = alpha
