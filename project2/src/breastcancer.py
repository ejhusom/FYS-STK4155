#!/usr/bin/env python3
# ============================================================================
# File:     breastcancer.py
# Author:   Erik Johannes Husom
# Created:  2019-11-01
# ----------------------------------------------------------------------------
# Description:
# Preprocess breast cancer data set.
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer



def visualize(df):
    plt.figure(figsize=(10,10))
    features = list(df.columns[1:10])
    ax = sns.heatmap(df[features].corr(), square=True, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

def sklearn_bunch_to_pandas_df(bunch):
    data = np.c_[bunch.data, bunch.target]
    columns = np.append(bunch.feature_names, ['target'])
    return pd.DataFrame(data, columns=columns)

def breast_cancer_dataset():
    # Reading data
    data = load_breast_cancer()
    y_names = data['target_names']
    y = data['target']
    X_names = data['feature_names']
    X = data['data']

    return X, y
