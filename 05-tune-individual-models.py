# !/usr/bin/env/ python3
# -*- coding: utf-8 -*-

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

import numpy as np
import pandas as pd
import pickle
from utility_functions import *

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, make_scorer

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# global variables
DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
# DATA_DIRECTORY = '//files/share/goods/OI Team'
os.chdir(DATA_DIRECTORY)

RANDOM_STATE = 5
FOLDS = 3
DEV_TEST_SIZE = .7
METRIC_NAMES = ['Precision', 'Recall', 'F1_score']
OFFER_PAIR_COLUMNS = ['offer_id_1', 'offer_id_2', 'filename', 'dataset', 'label', 'file_category']

# list of models to fit
MODELS = [RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', verbose=2),
          GradientBoostingClassifier(random_state=RANDOM_STATE, n_iter_no_change=30, verbose=2)]

model_names = [model.__class__.__name__ for model in MODELS]
model_dict = dict(zip(model_names, MODELS))

# set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 250)

# provide input file
input_file_name = 'attribute_comparison_features-7.csv' # input('Input the features file')
assert input_file_name in os.listdir(), 'An input file is missing'

# read input file
symbolic_similarity_features = reduce_mem_usage(pd.read_csv(input_file_name))
print(symbolic_similarity_features.columns.tolist())

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

# center and scale for SVM
scaler = StandardScaler()
symbolic_similarity_features = scaler.fit_transform(symbolic_similarity_features)

# n_features = symbolic_similarity_features.shape[1]

# train and test features
train_features, test_features = symbolic_similarity_features[train_indices, :],\
                                symbolic_similarity_features[test_indices, :]

dev_train_features, dev_test_features, dev_train_labels, dev_test_labels =\
    train_test_split(train_features, train_labels, test_size=DEV_TEST_SIZE, random_state=RANDOM_STATE)

print('Dev Train Feature Shape')
print(dev_train_features.shape)

rf_grid_params = {'max_depth': [None, 10],
                  'max_features': ['auto', None],
                  'max_leaf_nodes': [None],
                  'min_impurity_decrease': [0.0, .1],
                  'min_impurity_split': [None],
                  'min_samples_leaf': [1, 5],
                  'min_samples_split': [2, 5],
                  'min_weight_fraction_leaf': [0.0],
                  'n_estimators': [1000, 1500]}

count_cv_models(rf_grid_params, FOLDS, .21)

gbm_grid_params = {'learning_rate': [.01, .025],
                   'max_depth': [None, 10],
                   'max_features': ['auto', None],
                   'max_leaf_nodes': [None, 10],
                   'min_impurity_decrease': [0.0, .1],
                   'min_samples_leaf': [1, 5],
                   'min_samples_split': [2, 5],
                   'min_weight_fraction_leaf': [0.0],
                   'n_estimators': [150, 300, 450],
                   'subsample': [.15, .33],
                   'warm_start': [True, False]}

count_cv_models(gbm_grid_params, FOLDS, .07)

grid_param_list = [rf_grid_params, gbm_grid_params]
grid_param_dict = dict(zip(model_names, grid_param_list))

# create stratified folds
skf = StratifiedKFold(n_splits=FOLDS, random_state=RANDOM_STATE)

# output DF
test_metrics = []
dev_test_metrics = []
model_durations = []
best_params_list = []

# save diagnostics
test_predictions = []
class_probabilities_list = []
fit_models = []
classification_reports = []
confusion_matrices = []

for model_name, model in model_dict.items():

    print(model_name, model)

    start_time = datetime.now()
    model_params = grid_param_dict[model_name]
    full_fit_model = model

    if model_params is None:

        # no CV grid search for NB
        full_fit_model.fit(train_features, train_labels)

        fit_models.append(full_fit_model)
        best_params_list.append(None)

    else:

        cv_model = GridSearchCV(model, model_params, cv=skf, n_jobs=-1, verbose=2)
        cv_model.fit(dev_train_features, dev_train_labels)

        # get the best CV parameters
        best_params = cv_model.best_params_
        best_params_list.append(best_params)

        # make dev test predictions & calculate scores
        dev_test_pred = cv_model.predict(dev_test_features)
        dev_test_scores = calculate_scores(dev_test_labels, dev_test_pred)
        dev_test_metrics.append(dev_test_scores)
        print(dev_test_scores)

        print('fit model to full training set')
        full_fit_model.set_params(**best_params)
        full_fit_model.fit(train_features, train_labels)
        get_duration_hours(start_time)

    fit_models.append(full_fit_model)

    # make test predictions
    test_pred = full_fit_model.predict(test_features)
    # test_class_probabilities = full_fit_model.predict_proba(test_features)

    test_predictions.append(test_pred)
    # class_probabilities_list.append(test_class_probabilities)

    # get the classification report
    class_report = classification_report(test_labels, test_pred)
    classification_reports.append(class_report)
    print(class_report)

    # get confusion matrix
    confusion_df = pd.DataFrame(confusion_matrix(test_labels, test_pred),
                                columns=["Predicted Class " + str(class_name) for class_name in class_labels],
                                index=["Class " + str(class_name) for class_name in class_labels])
    confusion_matrices.append(confusion_df)
    print(confusion_df)

    # get scores
    model_metrics = calculate_scores(test_labels, test_pred)

    print(METRIC_NAMES)
    print(model_metrics)
    test_metrics.append(model_metrics)

    # get training duration
    hours = get_duration_hours(start_time)
    model_durations.append(hours)

print('Create Metrics DF')

sklearn_models_df = pd.DataFrame(test_metrics, columns=METRIC_NAMES, index=model_names)
sklearn_models_df['training_time'], sklearn_models_df['best_params'] = model_durations, best_params_list
print(sklearn_models_df.iloc[:, :4])

print('Save the results')

# create output directory if it doesn't exist
output_directory = input_file_name[: input_file_name.find('.csv')]
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# create output filename
output_file_name = output_directory + '-tuned-results.csv'

os.chdir(output_directory)
with open('sklearn_models.pkl', 'wb') as f:
    pickle.dump(fit_models, f)

with open('sklearn_test_predictions.pkl', 'wb') as f:
        pickle.dump(test_predictions, f)

with open('sklearn_class_probabilities.pkl', 'wb') as f:
    pickle.dump(class_probabilities_list, f)

with open('sklearn_confusion_matrices.pkl', 'wb') as f:
    pickle.dump(confusion_matrices, f)

# save the model metrics
sklearn_models_df.reset_index().to_csv(output_file_name, index=False)
