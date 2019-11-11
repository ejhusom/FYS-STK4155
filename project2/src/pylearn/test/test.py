import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pylearn.logisticregression import *


class Test():

    def __init__(self):
        pass


    def testLogisticRegression(self):
        dataset = datasets.load_breast_cancer()
        data = dataset.data
        target = dataset.target

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = linear_model.LogisticRegressionCV()
        model = SGDClassification()



        acc_model = np.zeros(10)
        acc_skl = np.zeros(10)


        for i in range(len(acc_model)):
            clf.fit(X_train_scaled, y_train)
            pred_skl = clf.predict(X_test_scaled)

            model.fit(X_train_scaled, y_train)
            pred_model = model.predict(X_test_scaled)

            acc_skl[i] = accuracy_score(pred_skl, y_test)
            acc_model[i] = accuracy_score(pred_model, y_test)

        print('scikit accuracy:', np.mean(acc_skl))
        print('pylearn accuracy:', np.mean(acc_model))
        print('diff:', abs(np.mean(acc_skl) - np.mean(acc_model)))


    def testNeuralNetwork(self):

        pass



if __name__ == '__main__':
    tests = Test()
    tests.testLogisticRegression()
