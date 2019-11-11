#!/usr/bin/env python3
# ============================================================================
# File:     metrics.py
# Created:  2019-10-04
# ----------------------------------------------------------------------------
# Description:
# Statistical metrics for machine learning.
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scikitplot.helpers import cumulative_gain_curve
from sklearn.metrics import auc
plt.style.use('ggplot')

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def bias(y_true, y_pred):
    return np.mean((y_true - np.mean(y_pred))**2)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2, axis=0)

def r2_score(y_true, y_pred):

    TSS = np.sum((y_true - np.mean(y_true))**2) # total sum of squares
    RSS = np.sum((y_true - y_pred)**2)          # residual sum of squares

    return 1 - RSS/TSS




def cumulative_gain_area_ratio(y_true, y_probas, onehot=False,
                            title='Cumulative Gains Curve',
                            ax=None, figsize=None, title_fontsize="large",
                            text_fontsize="large"):
    """
    Refactored code from scikit-plot's plot_cumulative_gain function.
    Plots the cumulative gain curve and calculates the area ratio.

    Inputs:
    - y_true: vector of targets (must be binary).
    - y_probas: probability of classification.
    - onehot: binary, True: y_vectors are of shape (n,2), False: shape (n,)
    """

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)
    classes = np.unique(y_true)

    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    #Workaround..
    if not onehot:
        y_probas = y_probas.reshape((len(y_probas), 1))
        y_probas = np.concatenate((np.zeros((len(y_probas), 1)), y_probas), axis=1)



    #Compute Cumulative Gain Curves
    percentages, gains = cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])


    #Calculate optimal model curve
    best_curve_x = [0,np.sum(y_true)/len(y_true), 1]
    best_curve_y = [0, 1, 1]


    #Calculate area ratio
    best_curve_area = auc(best_curve_x, best_curve_y) - 0.5
    model_curve_area = auc(percentages, gains) - 0.5
    area_ratio = model_curve_area/best_curve_area


    #plotting
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains, lw=2, label='Model')
    ax.plot(best_curve_x, best_curve_y, lw=2, label='Best curve')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.2])

    ax.set_xlabel('Percentage of data', fontsize=text_fontsize)
    ax.set_ylabel('Cumulative percentage of target data', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)
    plt.show()

    return area_ratio
