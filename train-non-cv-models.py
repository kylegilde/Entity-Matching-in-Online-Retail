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

import numpy as np
import pandas as pd
import pickle
from utility_functions import *

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, make_scorer

from sklearn.naive_bayes import GaussianNB #alpha smoothing?
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression  #LogisticRegression(random_state=0)

DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
DATA_DIRECTORY = '//files/share/goods/OI Team'
os.chdir(DATA_DIRECTORY)

RANDOM_STATE = 5
FOLDS = 2
DEV_TEST_SIZE = .95
# ALL_FEATURES = ['brand', 'manufacturer', 'gtin', 'mpn', 'sku', 'identifier', 'name', 'price', 'description'] # 'category'
OFFER_PAIR_COLUMNS = ['offer_id_1', 'offer_id_2', 'filename', 'dataset', 'label', 'file_category']

# list of models to fit
MODELS = [GaussianNB(),
          SVC(random_state=RANDOM_STATE, class_weight='balanced', probability=True, cache_size=1000, verbose=2),
          RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', verbose=2),
          GradientBoostingClassifier(random_state=RANDOM_STATE, n_iter_no_change=30, verbose=2)]

# SVC notes: https://scikit-learn.org/stable/modules/svm.html#complexity

MODEL_NAMES =['Naive Bayes', 'SVM', 'Random Forest', 'Gradient Boosting']

# list of scoring metrics
SCORERS = {'Precision': make_scorer(precision_score),
           'Recall': make_scorer(recall_score),
           'F1_score': make_scorer(f1_score)}

METRIC_NAMES = SCORERS.keys()

# provide input file
input_file_name = input('Input the features file')
assert input_file_name in os.listdir(), 'An input file is missing'

# read input file
symbolic_similarity_features = reduce_mem_usage(pd.read_csv(input_file_name))
print(symbolic_similarity_features.columns.tolist())

# create output directory if it doesn't exist
# output_directory = input_file_name[: input_file_name.find('.csv')]
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
#
# # create output filename
# output_file_name = output_directory + '-results.csv'

# get the train & test indices
train_indices, test_indices = symbolic_similarity_features.dataset.astype('object').apply(lambda x: x == 'train').values,\
                              symbolic_similarity_features.dataset.astype('object').apply(lambda x: x == 'test').values

# get the labels
all_labels = symbolic_similarity_features.label
train_labels, test_labels = all_labels[train_indices], all_labels[test_indices]
class_labels = np.sort(all_labels.unique())

# create features df
symbolic_similarity_features.set_index(OFFER_PAIR_COLUMNS, inplace=True)

print(symbolic_similarity_features.columns.tolist())
print(symbolic_similarity_features.info())
print(symbolic_similarity_features.shape)
print(symbolic_similarity_features.describe())

# train and test features
train_features, test_features = symbolic_similarity_features.loc[train_indices, :],\
                                symbolic_similarity_features.loc[test_indices, :]


dev_train_features, dev_test_features, dev_train_labels, dev_test_labels =\
    train_test_split(train_features, train_labels, test_size=DEV_TEST_SIZE, random_state=RANDOM_STATE)

print('Dev Train Feature Shape')
print(dev_train_features.shape)

# output DF
test_metrics = []
model_durations = []
best_params_list = []

# save diagnostics
test_predictions = []
class_probabilities_list = []
fit_models = []
classification_reports = []
confusion_matrices = []

for i, model in enumerate(MODELS):
    model_name = model.__class__.__name__
    print(model_name)

    model.fit(train_features, train_labels)
    test_pred = model.predict(test_features)
    test_predictions.append(test_pred)

    # get scores
    model_metrics = [precision_score(test_labels, test_pred),
                     recall_score(test_labels, test_pred),
                     f1_score(test_labels, test_pred)]

    print(METRIC_NAMES)
    print(model_metrics)
    test_metrics.append(model_metrics)

    fit_models.append(cv_model)

    # get training duration
    hours = get_duration_hours(start_time)
    model_durations.append(hours)