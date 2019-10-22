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
from logisticregression import SGDClassification



def scale(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test


def visualize(df):

    plt.figure(figsize=(10,10))

    features = list(df.columns[1:10])

    ax = sns.heatmap(df[features].corr(), square=True, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

 



def bunch2dataframe(bunch):

    data = np.c_[bunch.data, bunch.target]
    columns = np.append(bunch.feature_names, ['target'])

    return pd.DataFrame(data, columns=columns)



def breast_cancer_analysis():

    # Reading data
    data = load_breast_cancer()
    y_names = data['target_names']
    y = data['target']
    X_names = data['feature_names']
    X = data['data']

    # Visualization of data
#    visualize(bunch2dataframe(data))


    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)


    # Normalizing data
    X_train, X_test = scale(X_train, X_test)


    # Prediction with Scikit-learn
    clf_skl = SGDClassifier()
    clf_skl.fit(X_train, y_train)
    y_pred_skl = clf_skl.predict(X_test)
    score_skl = accuracy_score(y_test, y_pred_skl)
    print(score_skl)
    print(clf_skl.coef_)


    # Creating logistic regression model
    clf = SGDClassification()
    beta = clf.fit(X_train, y_train, batch_size=10, n_epochs=100)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
    print(beta)


if __name__ == '__main__':
    np.random.seed(2019)
    breast_cancer_analysis()
