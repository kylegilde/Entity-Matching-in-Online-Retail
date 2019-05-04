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

import numpy as np
import pandas as pd
import pickle
# from json_parsing_functions import *
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
FOLDS = 5
ALL_FEATURES = ['brand', 'manufacturer', 'gtin', 'mpn', 'sku', 'identifier', 'name', 'category', 'price', 'description']

SCORERS = {'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}

assert 'symbolic_similarity_features.csv' in os.listdir(), 'An input file is missing'

symbolic_similarity_features = reduce_mem_usage(pd.read_csv('symbolic_similarity_features.csv'))

# get the train & test indices
train_indices, test_indices = symbolic_similarity_features.dataset.astype('object').apply(lambda x: x == 'train'),\
                              symbolic_similarity_features.dataset.astype('object').apply(lambda x: x == 'test')

# get the labels
all_labels = symbolic_similarity_features.label
train_labels, test_labels = all_labels[train_indices], all_labels[test_indices]
class_labels = np.sort(all_labels.unique())

# train and test features
train_features, test_features = symbolic_similarity_features.loc[train_indices, ALL_FEATURES],\
                                symbolic_similarity_features.loc[test_indices, ALL_FEATURES]

# dev_train_features, dev_test_features, dev_train_labels, dev_test_labels = \
#     train_test_split(train_features, train_labels, test_size=0.2, stratify=train_labels)


MODELS = [GaussianNB(),
          SVC(random_state=RANDOM_STATE, class_weight='balanced', probability=True, cache_size=1000, verbose=2),
          RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', verbose=2),
          GradientBoostingClassifier(random_state=RANDOM_STATE, verbose=2)]

svc_grid_params = {'C': [0.001, 0.01, 0.1, 1, 10],
                   'gamma': [0.001, 0.01, 0.1, 1]}

rf_grid_params = {'max_depth': [None, 5, 10, 20],
                  'max_features': ['auto', None],
                  'max_leaf_nodes': [None],
                  'min_impurity_decrease': [0.0],
                  'min_impurity_split': [None],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'min_weight_fraction_leaf': [0.0],
                  'n_estimators': [100, 500, 1000, 2000]}

gbm_grid_params = {'learning_rate': [.025, .05, .075, .1],
                   'max_depth': [None, 5, 10, 20],
                   'max_features': ['auto'],
                   'max_leaf_nodes': [None],
                   'min_impurity_decrease': [0.0],
                   'min_impurity_split': [None],
                   'min_samples_leaf': [1],
                   'min_samples_split': [2],
                   'min_weight_fraction_leaf': [0.0],
                   'n_estimators': [200, 300, 400],
                   'subsample': [.1, .25, .5, .75]}

grid_param_list = [None, svc_grid_params, rf_grid_params, gbm_grid_params]

# for i in range(len(MODELS)):
#     GridSearchCV(MODELS[i], grid_param_list[i], cv=skf, n_jobs=-1, verbose=2)

METRIC_NAMES = ['precision', 'recall', 'f1']
skf = StratifiedKFold(n_splits=FOLDS, random_state=RANDOM_STATE)

# output DF
model_names = []
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
    model = MODELS[3]
    start_time = datetime.now()

    model_name = model.__class__.__name__
    model_names.append(model_name)
    print(model_name)

    if model_name == 'GaussianNB':

        # no CV grid search for NB
        cv_model = model
        cv_model.fit(train_features, train_labels)

        fit_models.append(cv_model)
        best_params_list.append(None)
    else:
        cv_model = GridSearchCV(model, grid_param_list[i], cv=skf, n_jobs=-1, verbose=2)
        cv_model.fit(train_features, train_labels)

        fit_models.append(cv_model)

        # get the best CV parameters
        best_params = cv_model.best_params_
        best_params_list.append(best_params)

    # make predictions
    test_pred, test_class_probabilities = cv_model.predict(test_features), \
                                          cv_model.predict_proba(test_features)

    test_predictions.append(test_pred)
    class_probabilities_list.append(test_class_probabilities)
    # test_class_probabilities = model.predict_proba(test_features)

    # get the classification report
    class_report = classification_report(test_labels, test_pred)
    classification_reports.append(class_report)
    print(class_report)

    # get confusion matrix
    confusion_df = pd.DataFrame(confusion_matrix(test_labels, test_pred),
                                columns = ["Predicted Class " + str(class_name) for class_name in class_labels],
                                index = ["Class " + str(class_name) for class_name in class_labels])
    confusion_matrices.append(confusion_df)
    print(confusion_df)

    # get scores
    model_metrics = [precision_score(test_labels, test_pred),
                     recall_score(test_labels, test_pred),
                     f1_score(test_labels, test_pred)]

    print(METRIC_NAMES)
    print(model_metrics)
    test_metrics.append(model_metrics)

    # get training duration
    hours = get_duration_hours(start_time)
    model_durations.append(hours)

sklearn_models_df = pd.DataFrame(test_metrics, columns=METRIC_NAMES, index=model_names)
sklearn_models_df['training_time'], sklearn_models_df['best_params'] = model_durations, best_params_list

print('Save the results')

with open('sklearn_models.pkl', 'wb') as f:
    pickle.dump(fit_models, f)

with open('sklearn_test_predictions.pkl', 'wb') as f:
        pickle.dump(test_predictions, f)

with open('sklearn_class_probabilities.pkl', 'wb') as f:
    pickle.dump(class_probabilities_list, f)

with open('sklearn_confusion_matrices.pkl', 'wb') as f:
    pickle.dump(confusion_matrices, f)

# save the model metrics
sklearn_models_df.reset_index().to_csv('sklearn_models_df.csv', index=False)




# non-CV version
# for i, model in enumerate(MODELS):
#     model_name = model.__class__.__name__
#     model_names.append(model_name)
#     print(model_name)
#
#     model.fit(train_features, train_labels)
#     test_pred = model.predict(test_features)
#     # test_predictions1.append(test_pred)
#     # get scores
#     model_metrics = [precision_score(test_labels, test_pred),
#                      recall_score(test_labels, test_pred),
#                      f1_score(test_labels, test_pred)]
#
#     print(METRIC_NAMES)
#     print(model_metrics)
#     test_metrics.append(model_metrics)