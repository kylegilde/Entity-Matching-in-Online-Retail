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

from json_parsing_functions import *

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# initialize constants
DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
os.chdir(DATA_DIRECTORY)
MAX_SVD_COMPONENTS = 1000
N_ROWS_PER_ITERATION = 2000

SHORT_TEXT_FEATURES = ['brand', 'manufacturer']
IDENTIFIER_FEATURES = ['gtin', 'mpn', 'sku', 'identifier']
ALL_SHORT_TEXT_FEATURES = SHORT_TEXT_FEATURES + IDENTIFIER_FEATURES + ['name']
LONG_TEXT_FEATURES = ['description']
STRONGLY_TYPED_FEATURES = ['category']
NUMERIC_FEATURE = ['price']

def levenshtein_similarity(str1, str2):
    """

    :param s1:
    :param s2:
    :return:
    """
    if pd.isnull(str1) or pd.isnull(str2):
        return 0
    else:
        return nltk.edit_distance(str1, str2) / np.max(len(str1), len(str2))




# load files
if 'train_test_stemmed_features.csv' in os.listdir() \
        and 'train_test_feature_pairs.csv' in os.listdir()\
        and 'train_test_df.csv' in os.listdir():

    train_test_stemmed_features = reduce_mem_usage(pd.read_csv('train_test_stemmed_features.csv'))\
        .set_index('offer_id')

    train_test_feature_pairs = reduce_mem_usage(pd.read_csv('train_test_feature_pairs.csv'))
    train_test_df = reduce_mem_usage(pd.read_csv('train_test_df.csv'))

    left_side_offer_ids, right_side_offer_ids = train_test_df[['offer_id_1']].set_index('offer_id_1'),\
                                                train_test_df[['offer_id_2']].set_index('offer_id_2')

    left_side_features, right_side_features = train_test_stemmed_features.join(left_side_offer_ids, how='inner').reset_index(drop=True),\
                                              train_test_stemmed_features.join(right_side_offer_ids, how='inner').reset_index(drop=True)

    print(train_test_stemmed_features.info(memory_usage='deep'))

    symbolic_similarity_features = pd.DataFrame()

    for column in left_side_offer_ids.columns:
        if column in ALL_SHORT_TEXT_FEATURES:
            # column = 'brand
            both_features = pd.concat([left_side_features[[column]], right_side_features[[column]]],\
                                      axis=1)

            both_features = pd.join([left_side_features[column], right_side_features[column])
            
            both_features.info()
            symbolic_similarity_features[column] = both_features\
                .apply(lambda df: levenshtein_similarity(df.iloc[:, 0], df.iloc[:, 1]))

        both_features.iloc[:, 1]
        elif column in LONG_TEXT_FEATURES:
            # column, feature_1, feature_2 = 'name', 'name_1', 'name_2'
            # column, feature_1, feature_2 = 'description', 'description_1', 'description_2'
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))

            n_rows = len(train_test_feature_pairs)
            text_data = pd.concat([train_test_feature_pairs[feature_1], train_test_feature_pairs[feature_2]]).fillna('')
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

            symbolic_similarity_features[column] = pd.concat(distances_list, ignore_index=True)
        elif column not in symbolic_similarity_features.columns:
            symbolic_similarity_features[column] = pd.Series(train_test_feature_pairs[feature_1].astype('object')\
                                                                     == train_test_feature_pairs[feature_2].astype('object')).astype('int8')

    symbolic_similarity_features.info(memory_usage='deep')





    # create feature variables
    # features_regex_1, features_regex_2 = r'(' + '|'.join(COLUMNS_TO_NORMALIZE) + ')_1',\
    #                                      r'(' + '|'.join(COLUMNS_TO_NORMALIZE) + ')_2'
    #
    #
    # features_1 = train_test_feature_pairs.columns[train_test_feature_pairs.columns.str.match(features_regex_1)]
    # features_2 = train_test_feature_pairs.columns[train_test_feature_pairs.columns.str.match(features_regex_2)]
    #
    # feature_dtypes = train_test_feature_pairs.dtypes.astype('str')[train_test_feature_pairs.columns.str.match(features_regex_1)]
    # for feature_dtype, column, feature_1, feature_2 in zip(feature_dtypes, COLUMNS_TO_NORMALIZE, features_1, features_2):
    #     print(feature_dtype, column, feature_1, feature_2)
