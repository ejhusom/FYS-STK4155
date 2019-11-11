#!/usr/bin/env python3
# ============================================================================
# File:     nn_classification.py
# Created:  2019-11-06
# ----------------------------------------------------------------------------
# Description:
# Analyze performance of neural network applied to classification problems.
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import time

import sklearn as skl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from scikitplot.metrics import plot_confusion_matrix
import scikitplot.metrics as skplt

from pylearn.logisticregression import SGDClassification
from pylearn.multilayerperceptron import MultilayerPerceptron
from pylearn.metrics import cumulative_gain_area_ratio


from breastcancer import *
from creditcard import *
from franke import *


def scale_data(train_data, test_data, scaler='standard'):
    """Scale train and test data.

    Parameters
    ----------
    train_data : array
        Train data to be scaled. Used as scale reference for test data.
    test_data : array
        Test data too be scaled, with train scaling as reference.
    scaler : str, default='standard'
        Options: 'standard, 'minmax'.
        Specifies whether to use sklearn's StandardScaler or MinMaxScaler.


    Returns
    -------
    train_data : array
        Scaled train data.
    test_data : array
        Scaled test data.

    """

    if scaler == 'standard':
        sc = StandardScaler()
    elif scaler == 'minmax':
        sc = MinMaxScaler()
    else:
        print('Scaler must be "standard" or "minmax"!')
        return None

    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    return train_data, test_data

def logistic_analysis(X, y):
    """Performing logistic regression, with cross-validation, using both
    Scikit-learn and pylearn.

    Parameters
    ----------
    X : array
        Design matrix.
    y : array
        Target vector.

    Returns
    -------
    Nothing.

    """

    print(CV(X, y, SGDClassifier(), n_splits=10))       # sklearn
    print(CV(X, y, SGDClassification(), n_splits=10))   # pylearn


def nn_classification_simple():
    """Perform simple classification case on Scikit-Learn's breast cancer data
    set. Function is used for testing.
    """

    # Loading, splitting and scaling data
    X, y = breast_cancer_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)
    X_train, X_test = scale_data(X_train, X_test, scaler='minmax')

    # One hot encoding targets
    y_train = y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()
    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()


    model = MultilayerPerceptron(
            hidden_layer_sizes=[10,10,10],
            eta=0.1, 
            learning_rate='constant',
            alpha=0.0,
            batch_size=100,
            n_epochs=200,
            weights_init='normal',
            act_func_str='sigmoid',
            output_func_str='sigmoid',
            cost_func_str='crossentropy')

    model.fit(X_train, y_train_1hot)
    y_pred = model.predict_class(X_test)
    print(f'pylearn accuracy: {accuracy_score(y_test, y_pred)}')
    nn_classification_plot(y_test, y_pred)


def nn_classification_analysis(train=False, options=[1, True]):
    """Analyzing behaviour of the accuracy score of a model based on which
    learning parameter (eta) is used, and how many epochs we run the training.

    Parameters
    ----------
    train : boolean
        If True: Model is trained, and then used for results.
        If False: Model is assumed to be already trained, and stored arrays are
        used for producing results. The stored arrays needs to be present in
        the same directory is this function.
    options : list
        options[0] : int, 1 or 2
            If 1, only gender, education and marital status are onehot encoded
            features. If 2, payment history is also onehot encoded.
        options[1] : boolean
            If False, unbalanced data set is used. If True, the data set is
            balanced.

    Returns
    -------
    Nothing.

    """

    if options[0] == 1:
        onehot_str = 'case1'
    else:
        onehot_str = 'case2'
    if options[1] == True:
        balance_str = 'balanced'
    else:
        balance_str = 'unbalanced'


    X, X_train, X_test, y, y_train, y_train_1hot, y_test, y_test_1hot = \
        credit_card_train_test('../data/credit_card.xls',
                which_onehot=options[0],
                balance_outcomes=options[1])

    # Test cases
    etas = np.logspace(-1, -4, 4)                 # 0.1, 0.01, ...
    n_epochs = [10, 100, 250, 500, 1000, 2000]             
    layers = [100,100,100]


    # Analyze accuracy score as function of learning rate
    accuracy_eta = np.zeros(len(etas))
    if train:
        i = 0
        for eta in etas:

            model = MultilayerPerceptron(
                        hidden_layer_sizes=layers,
                        eta=eta, 
                        alpha=0.0, 
                        batch_size=100,
                        learning_rate='constant',
                        n_epochs=100, 
                        weights_init='normal',
                        act_func_str='sigmoid',
                        cost_func_str='crossentropy',
                        output_func_str='sigmoid')

            model.fit(X_train, y_train_1hot)
            y_pred_test = model.predict_class(X_test)
            accuracy_eta[i] = accuracy_score(y_test, y_pred_test)
            print(f'Accuracy: {accuracy_eta[i]}')
            i += 1
            print(f'Eta={eta} done')


        # Finding the optimal learning rate eta
        eta_opt = etas[np.argmax(accuracy_eta)]
        print(eta_opt)
        print(f'Optimal eta: {eta_opt}')

        # Saving the accuracy. Optional: With timestring in filename.
        # timestr = time.strftime('%Y%m%d-%H%M')
        # np.save(timestr + '-accuracy_eta', accuracy_eta)
        np.save(f'class_accuracy_eta_o{options[0]}_b{options[1]}', accuracy_eta)

        # Analyze accuracy score as function of number of epochs
        accuracy_epoch = np.zeros(len(n_epochs))
        i = 0
        for n in n_epochs:

            model = MultilayerPerceptron(
                        hidden_layer_sizes=layers,
                        eta=eta_opt, 
                        alpha=0.0, 
                        batch_size=100,
                        learning_rate='constant',
                        n_epochs=n, 
                        weights_init='normal',
                        act_func_str='sigmoid',
                        cost_func_str='crossentropy',
                        output_func_str='sigmoid')

            model.fit(X_train, y_train_1hot)
            y_pred_test = model.predict_class(X_test)
            accuracy_epoch[i] = accuracy_score(y_test, y_pred_test)
            print(f'Accuracy: {accuracy_epoch[i]}')
            i += 1
            print(f'Epochs={n} done')

        np.save(f'class_accuracy_epoch_o{options[0]}_b{options[1]}', accuracy_epoch)
        np.save(f'class_y_pred_test_maxepoch_o{options[0]}_b{options[1]}', y_pred_test)

    # Loading accuracy scores
    accuracy_eta = np.load(f'class_accuracy_eta_o{options[0]}_b{options[1]}.npy')
    accuracy_epoch = np.load(f'class_accuracy_epoch_o{options[0]}_b{options[1]}.npy')
    

    # Plotting the analysis
    fig = plt.figure(figsize=(9.5,4.5))

    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(r'$\log_{10}$ of Learning rate')
    ax1.set_ylabel('Accuracy score')
    ax1.plot(np.log10(etas), accuracy_eta, '.-', label= onehot_str + ', ' +
            balance_str)
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('number of epochs')
    ax2.plot(n_epochs, accuracy_epoch, '.-', label=onehot_str + ', ' +
            balance_str)
    ax2.legend()

    plt.savefig(f'class-eta-accuracy_o{options[0]}_b{options[1]}.pdf')
    
    return eta_opt, accuracy_epoch[-1]


def nn_classification_plot_analysis(all_options):
    """Plotting the full analysis when running nn_classification_analysis() for
    all four possible versions of the credit card data set. This function
    assumes that nn_classification_analysis() has been run, and that the arrays
    stored in that function is still present in this directory.

    Parameters
    ----------
    all_options : nested list
        List of all the options that define a given data set. See the function
        nn_classification_analysis() for how options are defined. One entry in
        all_options should contain another list, which is of the same format as
        the 'options'-parameter in nn_classification_analysis.

    Returns
    -------
    Nothing.

    """

    fig = plt.figure(figsize=(9.5,4.5))

    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(r'$\log_{10}$ of Learning rate')
    ax1.set_ylabel('Accuracy score')
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('number of epochs')

    for options in all_options:
        if options[0] == 1 and not options[1]:
            label_num = '1'
        elif options[0] == 2 and not options[1]:
            label_num = '2'
        elif options[0] == 1 and options[1]:
            label_num = '3'
        elif options[0] == 2 and options[1]:
            label_num = '4'

        label = 'preproc. method ' + label_num
        
        etas = np.logspace(-1, -4, 4)                 # 0.1, 0.01, ...
        n_epochs = [10, 100, 250, 500, 1000, 2000]             
        accuracy_eta = np.load(f'class_accuracy_eta_o{options[0]}_b{options[1]}.npy')
        accuracy_epoch = np.load(f'class_accuracy_epoch_o{options[0]}_b{options[1]}.npy')

        ax1.plot(np.log10(etas), accuracy_eta, '.-', label=label)
        ax2.plot(n_epochs, accuracy_epoch, '.-', label=label)

    ax1.legend()
    ax2.legend()
    plt.savefig('class-eta-accuracy_full_analysis.pdf')


def nn_classification_heatmap(train=False, options=[1, False], eta=0.1):
    """Grid search for optimal hidden layer configuration in neural network.

    Parameters
    ----------
    train : boolean
        If True: Model is trained, and then used for results.
        If False: Model is assumed to be already trained, and stored arrays are
        used for producing results. The stored arrays needs to be present in
        the same directory is this function.
    options : list
        options[0] : int, 1 or 2
            If 1, only gender, education and marital status are onehot encoded
            features. If 2, payment history is also onehot encoded.
        options[1] : boolean
            If False, unbalanced data set is used. If True, the data set is
            balanced.
    eta : float, default=0.1
        Learning rate.

    Returns
    -------
    layers_opt : int
        The number of layers which gave the highest accuracy score (dependent
        on the number of nodes).
    nodes_opt : int
        The number of nodes which gave the highest accuracy score (dependent on
        the number of layers).

    """

    X, X_train, X_test, y, y_train, y_train_1hot, y_test, y_test_1hot = \
        credit_card_train_test('../data/credit_card.xls',
                which_onehot=options[0],
                balance_outcomes=options[1])

    # Test cases
    n_layers = np.arange(1, 5, 1)                   # 1, 2, 3, ...
    n_nodes = np.arange(40, 121, 20)                # 10, 20, 30, ...

    accuracy = np.zeros((len(n_layers), len(n_nodes)))

    if train:
        i = 0
        for l in n_layers:
            j = 0
            for n in n_nodes:
                layers = list(np.ones(l, dtype='int') * n)

                model = MultilayerPerceptron(
                            hidden_layer_sizes=layers,
                            eta=eta, 
                            alpha=0.0, 
                            batch_size=100,
                            learning_rate='constant',
                            n_epochs=100, 
                            weights_init='normal',
                            act_func_str='sigmoid',
                            cost_func_str='crossentropy',
                            output_func_str='sigmoid')

                model.fit(X_train, y_train_1hot)
                y_pred_test = model.predict_class(X_test)
                accuracy[i,j] = accuracy_score(y_test, y_pred_test)
                print(f'Accuracy: {accuracy[i,j]}')
                j += 1
                print(f'Nodes done: {n}')
            i += 1
            print(f'Layers done: {l}')

        np.save('accuracy_heat', accuracy)

    accuracy = np.load('accuracy_heat.npy')

    # Finding the configuration that gave the highest accuracy, and storing the
    # result.
    max_idcs = np.where(accuracy == np.max(accuracy))
    accuracy_opt = np.max(accuracy)
    layers_opt = n_layers[max_idcs[0]]
    nodes_opt = n_nodes[max_idcs[1]]

    print(f'Best accuracy: {accuracy_opt}')
    print(f'Best number of layers: {layers_opt}')
    print(f'Best number of nodes: {nodes_opt}')

    plt.figure(figsize=(9.5,4.5))

    ax = sns.heatmap(accuracy, annot=True, xticklabels=n_nodes, yticklabels=n_layers)
    ax.add_patch(Rectangle((max_idcs[1], max_idcs[0]), 1, 1, fill=False, edgecolor='red', lw=3))
    ax.set_xlabel('Number of nodes per layer')
    ax.set_ylabel('Number of layers')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('class_heatmap.pdf')

    return layers_opt, nodes_opt


def nn_classification_optimal(train=False, options=[2, True], eta=0.1,
        layers=3, nodes=80):
    """Training a model with given parameters, with a high number of epochs.

    Parameters
    ----------
    train : boolean
        If True: Model is trained, and then used for results.
        If False: Model is assumed to be already trained, and stored arrays are
        used for producing results. The stored arrays needs to be present in
        the same directory is this function.
    options : list
        options[0] : int, 1 or 2
            If 1, only gender, education and marital status are onehot encoded
            features. If 2, payment history is also onehot encoded.
        options[1] : boolean
            If False, unbalanced data set is used. If True, the data set is
            balanced.
    eta : float, default=0.1
        Learning rate.
    layers : int
        Number of hidden layers.
    nodes : int
        Number of nodes in each hidden layer.

    Returns
    -------
    Nothing.

    """


    X, X_train, X_test, y, y_train, y_train_1hot, y_test, y_test_1hot = \
        credit_card_train_test('../data/credit_card.xls',
                which_onehot=options[0],
                balance_outcomes=options[1])

    hl = list(np.ones(layers, dtype='int') * int(nodes))
    print(f'Hidden layers: {hl}')

    if train:
        i = 0
        model = MultilayerPerceptron(
                    hidden_layer_sizes=hl,
                    eta=eta, 
                    alpha=0.0, 
                    batch_size=100,
                    learning_rate='constant',
                    n_epochs=10000, 
                    weights_init='normal',
                    act_func_str='sigmoid',
                    cost_func_str='crossentropy',
                    output_func_str='sigmoid')

        model.fit(X_train, y_train_1hot)

        np.save('weights', model.weights)
        np.save('bias', model.biases)

        y_pred = model.predict_class(X_test)
        y_pred_probas = model.predict(X_test)
        y_pred_train = model.predict_class(X_train)
        y_pred_train_probas = model.predict(X_train)

        np.save('class_X_train_optimal', X_train)
        np.save('class_X_test_optimal', X_test)
        np.save('class_y_test_optimal', y_test)
        np.save('class_y_pred_optimal', y_pred)
        np.save('class_y_pred_optimal_probas', y_pred_probas)
        np.save('class_y_train_optimal', y_train)
        np.save('class_y_train_1hot_optimal', y_train_1hot)
        np.save('class_y_pred_train_optimal', y_pred_train)
        np.save('class_y_pred_train_optimal_probas', y_pred_train_probas)

    y_test = np.load('class_y_test_optimal.npy')
    y_pred = np.load('class_y_pred_optimal.npy')
    y_pred_probas = np.load('class_y_pred_optimal_probas.npy')
    y_train = np.load('class_y_train_optimal.npy')
    y_pred_train = np.load('class_y_pred_train_optimal.npy')
    y_pred_train_probas = np.load('class_y_pred_train_optimal_probas.npy')
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f'Accuracy: {accuracy}')
    print(f'Accuracy train: {accuracy_train}')
    print(f'Gain: {cumulative_gain_area_ratio(y_test, y_pred_probas, onehot=True, text_fontsize="large")}')
    print(f'Gain train: {cumulative_gain_area_ratio(y_train, y_pred_train_probas, onehot=True, text_fontsize="large")}')
    nn_classification_plot(y_test, y_pred)



def nn_classification_skl():
    """Comparison of sklearn and pylearn on breast cancer data set."""

    X, y = breast_cancer_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
            random_state = 0)

    X_train, X_test = scale_data(X_train, X_test, scaler='minmax')
    
    y_train = y_train.reshape(-1,1)
    encoder = OneHotEncoder(categories='auto')
    y_train_1hot = encoder.fit_transform(y_train).toarray()
    y_test_1hot = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

    hl = [80,80,80]
    eta = 1e-1
    n_epochs = 1000

    model = MultilayerPerceptron(
                hidden_layer_sizes=hl,
                eta=eta, 
                alpha=0.0, 
                batch_size=100,
                learning_rate='constant',
                n_epochs=n_epochs, 
                weights_init='normal',
                act_func_str='sigmoid',
                cost_func_str='crossentropy',
                output_func_str='sigmoid')

    model.fit(X_train, y_train_1hot)

    y_pred_probas = model.predict(X_test)
    y_pred = model.predict_class(X_test)
    print(f'pylearn accuracy: {accuracy_score(y_test, y_pred)}')
    print(cumulative_gain_area_ratio(y_test, y_pred_probas, onehot=True))
    nn_classification_plot(y_test, y_pred)


    dnn = MLPClassifier(hidden_layer_sizes=hl, 
                        activation='logistic',
                        alpha=0.0, 
                        learning_rate_init=eta, 
                        max_iter=n_epochs,
                        batch_size=100, 
                        learning_rate='constant')
    dnn.fit(X_train, y_train_1hot)
    y_pred_probas = dnn.predict_proba(X_test)
    y_pred = np.argmax(y_pred_probas, axis=1)
    print(f'Scikit accuracy: {accuracy_score(y_test, y_pred)}')
    print(cumulative_gain_area_ratio(y_test, y_pred_probas, onehot=True))
    nn_classification_plot(y_test, y_pred)




def nn_classification_plot(y_test, y_pred):
    """Plotting confusion matrix of a classification model."""

    ax = plot_confusion_matrix(y_test, y_pred, normalize=True, cmap='Blues',
            title=' ')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('confusionmatrix.pdf')
    plt.show()




if __name__ == '__main__':
    pass

