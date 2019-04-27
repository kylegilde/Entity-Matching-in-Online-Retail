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
import nltk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from json_parsing_functions import *

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def levenshtein_similarity(df_row):
    """
    https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe/13337376
    :param s1:
    :param s2:
    :return:
    """

    str1, str2 = df_row[0], df_row[1]

    if pd.isnull(str1) or pd.isnull(str2):
        return 0
    else:
        return 1 - nltk.edit_distance(str1, str2) / max(len(str1), len(str2))

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

ALL_FEATURES = ALL_SHORT_TEXT_FEATURES + LONG_TEXT_FEATURES + STRONGLY_TYPED_FEATURES + NUMERIC_FEATURE


# load files
if 'train_test_stemmed_features.csv' in os.listdir() \
        and 'train_test_df.csv' in os.listdir():

    train_test_stemmed_features = reduce_mem_usage(pd.read_csv('train_test_stemmed_features.csv'))\
        .set_index('offer_id')

    train_test_df = reduce_mem_usage(pd.read_csv('train_test_df.csv'))

    left_side_offer_ids, right_side_offer_ids = train_test_df[['offer_id_1']].set_index('offer_id_1'),\
                                                train_test_df[['offer_id_2']].set_index('offer_id_2')

    left_side_features, right_side_features = train_test_stemmed_features.join(left_side_offer_ids, how='inner').reset_index(drop=True),\
                                              train_test_stemmed_features.join(right_side_offer_ids, how='inner').reset_index(drop=True)

    symbolic_similarity_features = pd.DataFrame(columns=left_side_features.columns)

    for column in ALL_FEATURES:

        both_features = left_side_features[[column]].join(right_side_features[[column]],
                                                  lsuffix="_1",
                                                  rsuffix="_2")

        if column in ALL_SHORT_TEXT_FEATURES:
            # column = 'brand'



            symbolic_similarity_features[column] = both_features.apply(levenshtein_similarity, axis=1)


        elif column in LONG_TEXT_FEATURES:

            column = 'description'
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))

            n_rows = len(train_test_feature_pairs)

            all_docs = np.stack(both_features)

            dtm = vectorizer.fit_transform(train_test_stemmed_features[column].fillna(''))

            # print(dtm.shape)
            # n_retained_components = min(dtm.shape[1] - 1, MAX_SVD_COMPONENTS)
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

        elif column in STRONGLY_TYPED_FEATURES:

            symbolic_similarity_features[column] = pd.Series(both_features.iloc[:, 0] == both_features.iloc[:, 0]).astype('int8')

        elif column in NUMERIC_FEATURE:

            symbolic_similarity_features[column] = \
                np.nan_to_num(np.absolute(both_features.iloc[:, 0] - both_features.iloc[:, 1]) / \
                              np.maximum(both_features.iloc[:, 0], both_features.iloc[:, 1]))





    symbolic_similarity_features.info(memory_usage='deep')



else:
    print("some input files not found")


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


# and 'train_test_feature_pairs.csv' in os.listdir()\
# train_test_feature_pairs = reduce_mem_usage(pd.read_csv('train_test_feature_pairs.csv'))
