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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
#from sklearn.metrics import accuracy
import sys

# Add machine learning methods to path
sys.path.append('./methods')

from Classification import *


def breast_cancer_analysis():

    # Reading data
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
    df['diagnosis'] = pd.Series(data=cancer.target, index=df.index)


    # Initial analysis of data
    print('Breast cancer data - info:')
    df.info()
    print(df.head(3))

    features_mean = list(df.columns[1:11])

    # Heat map
    plt.figure(figsize=(10,10))
    sns.heatmap(df[features_mean].corr(), annot=True, square=True,
                cmap='coolwarm')
    plt.show()

    # Scatter matrix
    color_dic = {'M':'red', 'B':'blue'}
    colors = df['diagnosis'].map(lambda x: color_dic.get(x))

    sm = pd.scatter_matrix(df[features_mean], c=colors, alpha=0.4,
            figsize=((15,15)))
    plt.show()

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
    #        random_state = 0)
    #sc = StandardScaler()



    # Creating logistic regression model
    #model = GradientDescent(mode='classification')
    #beta_SGD = model.SGD(X, y, batch_size=10)
    #Z = model.predict(np.c_[xx.ravel(), yy.ravel()], beta_SGD)


if __name__ == '__main__':
    breast_cancer_analysis()
