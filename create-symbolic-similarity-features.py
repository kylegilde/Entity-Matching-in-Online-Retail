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

import nltk
import numpy as np
import pandas as pd

from json_parsing_functions import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def levenshtein_similarity(df_row):
    """

    Calculates the scaled Levenshtein similarity for 2 strings.

    :param df_row: a list-like object containing 2 strings
    :return: a float between 0 and 1
    """

    str1, str2 = df_row[0], df_row[1]

    if pd.isnull(str1) or pd.isnull(str2):
        return 0
    else:
        return 1 - nltk.edit_distance(str1, str2) / max(len(str1), len(str2))

start = datetime.now()

# initialize constants
DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
os.chdir(DATA_DIRECTORY)
MAX_SVD_COMPONENTS = 3000
VARIANCE_EXPLAINED = 0.999
N_ROWS_PER_ITERATION = 2000

# initialize the variable types
SHORT_TEXT_FEATURES = ['brand', 'manufacturer']
IDENTIFIER_FEATURES = ['gtin', 'mpn', 'sku', 'identifier']
ALL_SHORT_TEXT_FEATURES = SHORT_TEXT_FEATURES + IDENTIFIER_FEATURES + ['name']

LONG_TEXT_FEATURES = ['description']

STRONGLY_TYPED_FEATURES = ['category']
NUMERIC_FEATURE = ['price']

ALL_FEATURES = ALL_SHORT_TEXT_FEATURES + STRONGLY_TYPED_FEATURES + NUMERIC_FEATURE + LONG_TEXT_FEATURES


# load files
if 'train_test_stemmed_features.csv' in os.listdir() \
        and 'train_test_df.csv' in os.listdir():

    train_test_stemmed_features = reduce_mem_usage(pd.read_csv('train_test_stemmed_features.csv'))\
        .set_index('offer_id')

    train_test_df = reduce_mem_usage(pd.read_csv('train_test_df.csv'))

    # create the left and right side features
    left_side_offer_ids, right_side_offer_ids = train_test_df[['offer_id_1']].set_index('offer_id_1'),\
                                                train_test_df[['offer_id_2']].set_index('offer_id_2')

    left_side_features, right_side_features = train_test_stemmed_features.join(left_side_offer_ids, how='inner').reset_index(drop=True),\
                                              train_test_stemmed_features.join(right_side_offer_ids, how='inner').reset_index(drop=True)

    symbolic_similarity_features = train_test_df.copy()

    del train_test_df
    gc.collect()

    for column in ALL_FEATURES:
        print('column:', column)
        print(datetime.now() - start)

        both_features = left_side_features[[column]].join(right_side_features[[column]],
                                                          lsuffix="_1",
                                                          rsuffix="_2")

        if column in ALL_SHORT_TEXT_FEATURES:

            symbolic_similarity_features[column] = both_features.apply(levenshtein_similarity, axis=1)

        elif column in STRONGLY_TYPED_FEATURES:

            symbolic_similarity_features[column] = pd.Series(both_features.iloc[:, 0]\
                                                             == both_features.iloc[:, 0]).astype('int8')

        elif column in NUMERIC_FEATURE:

            symbolic_similarity_features[column] = \
                np.nan_to_num(np.absolute(both_features.iloc[:, 0] - both_features.iloc[:, 1]) / \
                              np.maximum(both_features.iloc[:, 0], both_features.iloc[:, 1]))

        elif column in LONG_TEXT_FEATURES:

            column = 'description'
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))

            # create a document-term matrix
            dtm = vectorizer.fit_transform(train_test_stemmed_features[column].fillna(''))
            print('dtm dimensions:', dtm.shape)

            # use Truncated SVD to select a smaller number of important features
            svd_model = TruncatedSVD(n_components=MAX_SVD_COMPONENTS).fit(dtm)
            print(svd_model.explained_variance_ratio_.sum(), 'variance explained')

            n_features = sum(svd_model.explained_variance_ratio_.cumsum() <= VARIANCE_EXPLAINED)
            print(n_features, "features explain this much of the variance:", VARIANCE_EXPLAINED)

            # fit the svd model and convert to df
            dtm_svd = pd.DataFrame(svd_model.transform(dtm)[:, :n_features],
                                   index=train_test_stemmed_features.index)

            del dtm, both_features, svd_model, left_side_features, right_side_features
            gc.collect()

            print('post-SVD dtm dimensions:', dtm_svd.shape)
            print(dtm_svd.info(memory_usage='deep'))

            # create the left & right side DTMs
            left_side_dtm_svd, right_side_dtm_svd = left_side_offer_ids.join(dtm_svd, how='inner').values,\
                                                    right_side_offer_ids.join(dtm_svd, how='inner').values

            del dtm_svd, left_side_offer_ids, right_side_offer_ids
            gc.collect()

            # calculate the cosine similarites for each pair of docs
            symbolic_similarity_features[column] = np.maximum(cosine_similarity(left_side_dtm_svd, right_side_dtm_svd), 0)

    print("symbolic_similarity_features saved")
    symbolic_similarity_features.to_csv('symbolic_similarity_features.csv', index=False)

    print(datetime.now() - start)

else:

    print("input files not found")




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
#             svd_dtm_1, svd_dtm_2 = features_svd[: n_rows], features_svd[n_rows: ]
#
#             n_loops = n_rows // N_ROWS_PER_ITERATION + 1
#
#             distances_list = []
#             for i in range(1, n_loops + 1):
#                 # i = 1
#                 print(i)
#                 mn, mx = (i - 1) * N_ROWS_PER_ITERATION, i * N_ROWS_PER_ITERATION
#                 i_cosines = pairwise_cosine_dist_between_matrices(svd_dtm_1[mn:mx, ], svd_dtm_2[mn:mx, ])
#                 distances_list.append(i_cosines)
#
#             symbolic_similarity_features[column] = pd.concat(distances_list, ignore_index=True)
#
#             train_test_stemmed_features.join(left_side_offer_ids, how='inner').reset_index(drop=True)
#             n_loops = n_rows // N_ROWS_PER_ITERATION + 1
#
#             distances_list = []
#             for i in range(1, n_loops + 1):
#                 # i = 1
#                 print(i)
#                 mn, mx = (i - 1) * N_ROWS_PER_ITERATION, i * N_ROWS_PER_ITERATION
#                 i_cosines = pairwise_cosine_dist_between_matrices(svd_dtm_1[mn:mx, ], svd_dtm_2[mn:mx, ])
#                 distances_list.append(i_cosines)
#
#             symbolic_similarity_features[column] = pd.concat(distances_list, ignore_index=True)
