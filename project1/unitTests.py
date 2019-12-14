from RegressionMethods import *
from main import *
from Resampling import *


class UnitTests():
    """
    Tests the OLS and Ridge implementations in the RegressionMethods class, by
    comparing its results with sklearns equivalent methods.
    A test case is initialized automatically in the __init__ function.
    """

    def __init__(self):
        self.n = 100
        np.random.seed(10)
        degree = 5
        sigma = 0.3
        x, y = generate_mesh(self.n)
        z = frankie_function(x, y, self.n, sigma)
        self.z_flat = np.ravel(z)
        self.X = create_design_matrix(x, y, degree)
        self.tol = 1e-15

    def test_ols(self):

        #perform ols on the data with sklean
        clf = skl.LinearRegression().fit(self.X, self.z_flat)
        z_pred_skl = clf.predict(self.X)


        #perform ols with our implementation
        model = RegressionMethods('ols')
        model.fit(self.X, self.z_flat)
        z_pred_model = model.predict(self.X)

        #assert difference in our implementation and sklearns
        diff = mean_squared_error(z_pred_skl, z_pred_model)
        print("Test OLS:")
        print("Max diff: ", diff)
        assert diff < self.tol


    def test_ridge(self):

        #perform ridge on the data with sklearn
        alpha = 0.1
        clf = skl.Ridge(alpha = alpha, fit_intercept=False).fit(self.X, self.z_flat)
        z_pred_skl = clf.predict(self.X)

        #perform ridge on the data with our implementation
        model = RegressionMethods(method = 'ridge', alpha = alpha)
        model.fit(self.X, self.z_flat)
        z_pred_model = model.predict(self.X)

        #assert difference
        diff = mean_squared_error(z_pred_skl, z_pred_model)
        print("Test Ridge:")
        print("Max diff: ", diff)
        assert diff < self.tol


if __name__ == '__main__':
    tests = UnitTests()
    tests.test_ols()
    tests.test_ridge()
    print("All tests passed")
