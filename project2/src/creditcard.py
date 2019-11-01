#!/usr/bin/env python3
# ============================================================================
# File:     credit_card.py
# Author:   Erik Johannes Husom
# Created:  2019-10-10
# ----------------------------------------------------------------------------
# Description:
# 
# ============================================================================
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

def preprocess_creditcard_data(filename):

    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)


    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


    #find and remove outliers in the data
    outlier_gender1 = np.where(X[:,1] < 1)[0]
    outlier_gender2 = np.where(X[:,1] > 2)[0]

    outlier_education1 = np.where(X[:,2] < 1)[0]
    outlier_education2 = np.where(X[:,2] > 4)[0]

    outlier_marital1 = np.where(X[:,3] < 1)[0]
    outlier_marital2 = np.where(X[:,3] > 3)[0]

    inds = np.concatenate((outlier_gender1,
                           outlier_gender2,
                           outlier_education1,
                           outlier_education2,
                           outlier_marital1,
                           outlier_marital2))


    outlier_rows = np.unique(inds)

    X = np.delete(X, outlier_rows, axis=0)
    y = np.delete(y, outlier_rows, axis=0)


    #split data into categorical and continuous features
    categorical_inds = (1, 2, 3)
    continuous_inds = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    X_cat = X[:,categorical_inds]
    X_cont = X[:, continuous_inds]


    #onehot encode categorical data
    onehotencoder = OneHotEncoder(categories="auto", sparse=False)
    preprocessor = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                ('onehot', onehotencoder, list(range(X_cat.shape[1])))])

    X_cat = preprocessor.fit_transform(X_cat)

    #join categorical and continuous features
    X = np.concatenate((X_cont, X_cat), axis=1)

    cont_feat_inds = list(range(X_cont.shape[1]))

    return X, np.ravel(y), cont_feat_inds







if __name__ == '__main__':
    pass
