#!/usr/bin/env python3
# ============================================================================
# File:     analysis.py
# Author:   Erik Johannes Husom
# Created:  2019-10-10
# ----------------------------------------------------------------------------
# Description:
# 
# ============================================================================
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Add machine learning methods to path
sys.path.append('./methods')

from Classification import *

def read_data(filename):
    """
    Read data from file into dataframe. Filename must be given with relative
    path.
    """
    
    if filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filename, header=1, skiprows=0, index_col=0)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename, header=1, skiprows=0, index_col=0)
    else:
        print('File must be in .xls, .xlsx or .csv format!')
        sys.exit(1)

    print(df)
    return df


def preprocess_credit_data(dataframe):
    """Preprocessing of credit card data."""

    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

    # Categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories="auto")

    X = ColumnTransformer(
        [("", onehotencoder, [3]),],
        remainder="passthrough"
    ).fit_transform(X)



if __name__ == '__main__':
    df = read_data('./../data/credit_card.xls')
    #preprocess_credit_data(df)
