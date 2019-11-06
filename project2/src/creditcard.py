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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score


def preprocess_CC_data(filename, which_onehot = 1):

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
    if which_onehot==1:
        """
        only marriage, sex and education onehot encoded
        """
        categorical_inds = (1, 2, 3)
        continuous_inds = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


    elif which_onehot==2:
        """
        all categories onehot encoded
        """
        categorical_inds = (1, 2, 3, 5, 6, 7, 8, 9, 10)
        continuous_inds = (0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

    else:
        print('which_onehot must be specified as either 1 or 2')
        exit(0)

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

    print('preprocessing done')
    return X, np.ravel(y), cont_feat_inds



def credit_card_train_test(filename, which_onehot=1, balance_outcomes=True):

    X, y, scale_columns = preprocess_CC_data(filename,
            which_onehot=which_onehot)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


    #balance training set such that outcomes are 50/50
    if balance_outcomes:
        non_default_inds = np.where(y_train==0)[0]
        default_inds = np.where(y_train==1)[0]

        remove_size = len(non_default_inds) - len(default_inds)
        remove_inds = np.random.choice(non_default_inds, size=remove_size, replace=False)

        X_train = np.delete(X, remove_inds, axis=0)
        y_train = np.delete(y, remove_inds, axis=0)

    minmaxscaler = MinMaxScaler()
    scaler = ColumnTransformer(
                        remainder='passthrough',
                        transformers=[('minmaxscaler', minmaxscaler, scale_columns)])


    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()
    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()


    return X, X_train, X_test, y, y_train, y_train_1hot, y_test, y_test_1hot


if __name__ == '__main__':
    pass
