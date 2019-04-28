# !/usr/bin/env/python365
"""
Created on Apr 27, 2019
@author: Kyle Gilde

This script takes the outputs of create-symbolic-features.py
and parse-json-to-dfs.py.

It outputs a df that contains the similarity vectors of the offer pairs
in the test and training sets.

"""

import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd

from json_parsing_functions import reduce_mem_usage

import scipy.stats as stats

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB #alpha smoothing?
from sklearn.svm import SVC #
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression  #LogisticRegression(random_state=0)

DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
os.chdir(DATA_DIRECTORY)
RANDOM_STATE = 5
ALL_FEATURES = ['brand', 'manufacturer', 'gtin', 'mpn', 'sku', 'identifier', 'name', 'category', 'price', 'description']

SCORERS = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}


if 'symbolic_similarity_features.csv' in os.listdir():
    symbolic_similarity_features = reduce_mem_usage(pd.read_csv('symbolic_similarity_features.csv'))

    # get the train & test indices
    train_indices, test_indices = symbolic_similarity_features.dataset.astype('object').apply(lambda x: x == 'train'),\
                                  symbolic_similarity_features.dataset.astype('object').apply(lambda x: x == 'test')

    # get the labels
    all_labels = symbolic_similarity_features.label
    train_labels, test_labels = all_labels[train_indices], all_labels[test_indices]
    
    
    train_features, test_features = symbolic_similarity_features.loc[train_indices, ALL_FEATURES],\
                                    symbolic_similarity_features.loc[test_indices, ALL_FEATURES]



    from sklearn.model_selection import train_test_split

    FOLDS = 5
    MODELS = [GaussianNB(),
              SVC(random_state=RANDOM_STATE, class_weight='balanced'),
              RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
              GradientBoostingClassifier(random_state=RANDOM_STATE)]

    from pprint import pprint
    pprint(MODELS[2].get_params())

    svc_grid_params = {'C': [0.001, 0.01, 0.1, 1, 10],
                       'gamma': [0.001, 0.01, 0.1, 1]}

    grid_search = GridSearchCV(SVC(random_state=RANDOM_STATE), svc_grid_params, cv=FOLDS,
                               n_jobs = -1, verbose = 2)
    grid_search.fit(train_features, train_labels)
    grid_search.best_params_

    rf_grid_params = {'bootstrap': True,
                      'class_weight': 'balanced',
                      'criterion': 'gini',
                      'max_depth': None,
                      'max_features': ['auto', None],
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_impurity_split': None,
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10],
                      'min_weight_fraction_leaf': 0.0,
                      'n_estimators': [100, 500, 1000, 2000],
                      'n_jobs': -1,
                      'oob_score': False,
                      'random_state': RANDOM_STATE,
                      'verbose': 1,
                      'warm_start': False}




    test_predictions = []
    classification_reports = []
    confusion_matrices = []
    model_names = []
    test_metrics = []
    METRIC_NAMES = ['precision', 'recall', 'f1']


    for model in MODELS:

        model_name = model.__class__.__name__
        model_names.append(model_name)
        print(model_name)

        model.fit(train_features, train_labels)
        test_pred = model.predict(test_features)
        test_predictions.append(test_pred)

        # predict_proba
        class_report = classification_report(test_labels, test_pred)
        classification_reports.append(class_report)
        print(class_report)

        confusion_df = pd.DataFrame(confusion_matrix(test_labels, test_pred),
                                    columns = ["Predicted Class " + str(class_name) for class_name in [0, 1]],
                                    index = ["Class " + str(class_name) for class_name in [0, 1]])

        confusion_matrices.append(confusion_df)
        print(confusion_df)

        model_metrics = [precision_score(test_labels, test_pred),
                         recall_score(test_labels, test_pred),
                         f1_score(test_labels, test_pred)]

        print(METRIC_NAMES)
        print(model_metrics)
        test_metrics.append(model_metrics)


    no1 = pd.DataFrame(test_metrics, columns=METRIC_NAMES, index=[n.__class__.__name__ for n in MODELS])
    pre_weighting = no1.copy()

    def perform_cross_val(models, X_train, y_train, scoring, cv=FOLDS):
        """
        perform cross-validation model fitting
        and returns the results

        :param models: list of sci-kit learn model classes
        :param X_train: training data set
        :param y_train: response labels
        :param cv: # of folds
        :return: a df with the accuracies for each model and fold
        """
        entries = []
        for model in models:
          model_name = model.__class__.__name__
          accuracies = cross_val_score(model, X_train, y_train, scoring=SCORERS, cv=cv)
          for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))

        return pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])




    def plot_cv_results(cv_df):
        """
        Plots the distributions of the accuracy SCORERS
        for each fold in each of our models.

        :param cv_df: the output df from perform_cross_val
        :return: None
        """
        plt.figure(figsize=[10, 10])
        sns.boxplot(x='model_name', y='accuracy', data=cv_df)
        f = sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
        f.xaxis.tick_top() # x labels on top
        plt.setp(f.get_xticklabels(), rotation=30, fontsize=20) # rotate and increase x labels
        plt.show()
        # calculate accuracy mean & std
        display(cv_df.groupby('model_name')\
                     .agg({"accuracy": [np.mean, stats.sem]})\
                     .sort_values(by=('accuracy', 'mean'), ascending=False)\
                     .round(4))




    X_train, X_dev_test, y_train, y_dev_test = train_test_split(train_features,
                                                                train_labels,\
                                                                test_size=.2,\
                                                                random_state = 0)

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)

    cv_scores = cross_val_score(models[0], X_train, y_train, SCORERS, cv=5)



    # run cv function
    results = perform_cross_val(models, X_train, y_train)

    plot_cv_results(results)

else:

    print("input files not found")

