# !/usr/bin/env/python365
"""
Created on Feb 23, 2019
@author: Kyle Gilde
"""
import os
import sys
from datetime import datetime

from urllib.parse import urlparse
import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utility_functions import *

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# initialize constants
DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
os.chdir(DATA_DIRECTORY)
MAX_SVD_COMPONENTS = 3000
N_ROWS_PER_ITERATION = 2000
comparison_features = ['brand', 'category', 'description', 'gtin', 'identifier', 'manufacturer',
                       'mpn', 'name', 'price', 'productID', 'sku']
large_text_features = ['name', 'description']


# set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

# user-defined functions
def pairwise_cosine_dist_between_matrices(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    cosine_matrix = np.dot(a, b.T) / \
                    np.dot(np.sqrt(np.dot(a, a.T).diagonal()),
                           np.sqrt(np.dot(b, b.T).diagonal()).T)
    # the truncated SVD creates some slightly negative values in the calculation
    # change these to zero
    return pd.Series(np.maximum(cosine_matrix.diagonal(), 0))

# load
if 'train_test_df_features.csv' in os.listdir():
    train_test_df_features = reduce_mem_usage(pd.read_csv('train_test_df_features.csv'))
    print(train_test_df_features.info(memory_usage='deep'))

    # get the train & test indices
    train_indices, test_indices = train_test_df_features.dataset.astype('object').apply(lambda x: x == 'train'),\
        train_test_df_features.dataset.astype('object').apply(lambda x: x == 'test')

    # get the labels
    all_labels = train_test_df_features.label
    train_labels, test_labels = all_labels[train_indices], all_labels[test_indices]

    # create feature variables
    features_regex_1, features_regex_2 = r'(' + '|'.join(comparison_features) + ')_1',\
                                         r'(' + '|'.join(comparison_features) + ')_2'


    features_1 = train_test_df_features.columns[train_test_df_features.columns.str.match(features_regex_1)]
    features_2 = train_test_df_features.columns[train_test_df_features.columns.str.match(features_regex_2)]

    feature_dtypes = train_test_df_features.dtypes.astype('str')[train_test_df_features.columns.str.match(features_regex_1)]

    distance_vector_features = pd.DataFrame()

    for feature_dtype, comparison_feature, feature_1, feature_2 in zip(feature_dtypes, comparison_features, features_1, features_2):
        print(feature_dtype, comparison_feature, feature_1, feature_2)

        if comparison_feature in large_text_features and comparison_feature not in distance_vector_features.columns:
            # comparison_feature, feature_1, feature_2 = 'name', 'name_1', 'name_2'
            # comparison_feature, feature_1, feature_2 = 'description', 'description_1', 'description_2'
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))

            n_rows = len(train_test_df_features)
            text_data = pd.concat([train_test_df_features[feature_1], train_test_df_features[feature_2]]).fillna('')
            dtm = vectorizer.fit_transform(text_data)

            print(dtm.shape)
            n_retained_components = min(dtm.shape[1] - 1, MAX_SVD_COMPONENTS)
            # SVD model
            svd_model = TruncatedSVD(n_components=n_retained_components).fit(dtm) 
            print(svd_model.explained_variance_ratio_.sum(), 'variance explained')

            # How many components explain 99% of the variance
            feature_threshold = sum(svd_model.explained_variance_ratio_.cumsum()[:MAX_SVD_COMPONENTS] <=.99)
            print(feature_threshold)

            # select the components that explain 99% of the variance
            features_svd = svd_model.transform(dtm)[:, :feature_threshold]
            print(features_svd.shape)
            #separate the features back into set 1 and 2
            svd_dtm_1, svd_dtm_2 = features_svd[: n_rows], features_svd[n_rows: ]

            n_loops = n_rows // N_ROWS_PER_ITERATION + 1

            distances_list = []
            for i in range(1, n_loops + 1):
                # i = 1
                print(i)
                mn, mx = (i - 1) * N_ROWS_PER_ITERATION, i * N_ROWS_PER_ITERATION
                i_cosines = pairwise_cosine_dist_between_matrices(svd_dtm_1[mn:mx, ], svd_dtm_2[mn:mx, ])
                distances_list.append(i_cosines)

            distance_vector_features[comparison_feature] = pd.concat(distances_list, ignore_index=True)
        elif comparison_feature not in distance_vector_features.columns:
            distance_vector_features[comparison_feature] = pd.Series(train_test_df_features[feature_1].astype('object')\
                                                                     == train_test_df_features[feature_2].astype('object')).astype('int8')

    distance_vector_features.info(memory_usage='deep')

    train_features, test_features = distance_vector_features.loc[train_indices, ],\
        distance_vector_features.loc[test_indices, :]

