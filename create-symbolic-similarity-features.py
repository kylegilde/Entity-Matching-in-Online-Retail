# !/usr/bin/env/python365
"""
Created on Apr 27, 2019
@author: Kyle Gilde

This script takes the outputs of create-symbolic-features.py
and parse-json-to-dfs.py.

It outputs a df that contains the similarity vectors of the offer pairs
in the test and training sets. The df is saved as symbolic_similarity_features.csv

"""

import os
import gc
from datetime import datetime

import nltk
import numpy as np
import pandas as pd

from json_parsing_functions import reduce_mem_usage


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine


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


def elementwise_cosine_similarity(df_row, n_features):
    """
    Calculates the elementwise cosine similarity with the apply method

    :param df_row: a DF row where the first n_features are the left side features
        and the last n_features are the right side features
    :param n_features: this is the number of features each side of the DTM has
    :return: float between 0 and 1
    """
    s1, s2 = df_row[:n_features], df_row[n_features:]

    if np.sum(s1) == 0 or np.sum(s2) == 0:
        return 0
    else:
        return cosine(s1, s2)

start = datetime.now()

# set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 0)

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
            # column = 'description'

            vectorizer = TfidfVectorizer(ngram_range=(1, 3))

            # create a document-term matrix
            dtm = vectorizer.fit_transform(train_test_stemmed_features[column].fillna(''))
            print('dtm dimensions:', dtm.shape)

            print('Use Truncated SVD to select a smaller number of important features')
            svd_model = TruncatedSVD(n_components=MAX_SVD_COMPONENTS).fit(dtm)
            print(svd_model.explained_variance_ratio_.sum(), 'variance explained')

            n_features = sum(svd_model.explained_variance_ratio_.cumsum() <= VARIANCE_EXPLAINED)
            print(n_features, "features explain this much of the variance:", VARIANCE_EXPLAINED)

            # fit the svd model and convert to df
            dtm_svd = pd.DataFrame(svd_model.transform(dtm)[:, :n_features],
                                   index=train_test_stemmed_features.index)

            print('post-SVD DTM dimensions:', dtm_svd.shape)
            print(dtm_svd.info(memory_usage='deep'))

            del dtm, both_features, svd_model, left_side_features, right_side_features
            gc.collect()

            # concatenate the left and right side DTMs
            both_sides_dtm_svd = pd.concat([left_side_offer_ids.join(dtm_svd, how='inner').reset_index(drop=True),
                                           right_side_offer_ids.join(dtm_svd, how='inner').reset_index(drop=True)],
                                           axis=1)

            del dtm_svd, left_side_offer_ids, right_side_offer_ids
            gc.collect()

            # calculate the cosine similarites for each pair of docs
            symbolic_similarity_features[column] = both_sides_dtm_svd.apply(elementwise_cosine_similarity,
                                                                            n_features=n_features,
                                                                            axis=1)

            symbolic_similarity_features.summary()

    print("symbolic_similarity_features saved")
    symbolic_similarity_features.to_csv('symbolic_similarity_features.csv', index=False)

    print(datetime.now() - start)

    symbolic_similarity_features.describe()

else:

    print("input files not found")
