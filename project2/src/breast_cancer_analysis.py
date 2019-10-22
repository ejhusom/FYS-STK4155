#!/usr/bin/env python3
# ============================================================================
# File:     breast_cancer_analysis.py
# Author:   Erik Johannes Husom
# Created:  2019-10-22
# ----------------------------------------------------------------------------
# Description:
# Analyze breast cancer data.
# ============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import sys

# Add machine learning methods to path
sys.path.append('./methods')

from Classification import *
from gradientdescent import GradientDescent


def breast_cancer_analysis():

    # Reading data
    data = load_breast_cancer()
    y_names = data['target_names']
    y = data['target']
    X_names = data['feature_names']
    X = data['data']


    # Prediction with Scikit-learn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)
    clf_skl = SGDClassifier()
    clf_skl.fit(X_train, y_train)
    y_pred = clf_skl.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)


    # Creating logistic regression model
    clf = GradientDescent(mode='classification', stochastic=True)
    clf.fit(X_train, y_train, batch_size=10)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)


if __name__ == '__main__':
    np.random.seed(2019)
    breast_cancer_analysis()
