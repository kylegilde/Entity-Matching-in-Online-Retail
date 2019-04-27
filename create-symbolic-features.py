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
MAX_SVD_COMPONENTS = 3000
N_ROWS_PER_ITERATION = 2000

TEXT_FEATURES = ['name', 'description']
SHORT_TEXT_FEATURES = ['brand', 'manufacturer']
IDENTIFIER_FEATURES = ['gtin', 'mpn', 'sku', 'identifier']
ALL_TEXT_FEATURES = TEXT_FEATURES + SHORT_TEXT_FEATURES + IDENTIFIER_FEATURES

STRONGLY_TYPED_FEATURES = ['category']
NUMERIC_FEATURE = ['price']

COLUMNS_TO_NORMALIZE = ALL_TEXT_FEATURES + STRONGLY_TYPED_FEATURES

COLUMNS_NEEDED = ['offer_id'] + COLUMNS_TO_NORMALIZE


# set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 0)

# load parsed features
if 'train_test_feature_pairs.csv' in os.listdir():
    train_test_offer_features = reduce_mem_usage(pd.read_csv('train_test_offer_features.csv'))\
        .set_index('offer_id')
    # the identifier and productID columns are mutually exclusively null
    train_test_offer_features['identifier'] = train_test_offer_features.identifier\
        .mask(pd.isnull, train_test_offer_features.productID)

train_test_normalized_features = train_test_offer_features[COLUMNS_TO_NORMALIZE]

# train_test_normalized_features.apply(pd.Series.nunique)
# train_test_normalized_features\
#     .melt()\
#     .dropna().groupby(['variable', 'value'])['value'].agg('count')\
#     .to_frame()\
#     .rename(index=str, columns={'value': 'value_counts'})\
#     .reset_index()\
#     .groupby(['variable'])\
#     .apply(lambda x: x.nlargest(10, 'value_counts'))\
#     .reset_index(drop=True)


import nltk
from nltk.corpus import stopwords
sb_stemmer = nltk.stem.SnowballStemmer('english')


def stem_and_remove_stopwords(doc):
    """
    Passed as the callable to Pandas.apply()
    Each token is stemmed and returned if it meets the conditions
    Sources: https://stackoverflow.com/a/36191362/4463701
    https://stackoverflow.com/a/51281480/4463701
    """
    stemmed_tokens = [sb_stemmer.stem(token) for token in nltk.word_tokenize(doc)\
                      if token not in stopwords.words('english')]

    return ' '.join(stemmed_tokens)

def clean_text(series):
    """

    1. remove html tags
    2. converts to lowercase
    3. replace 2 consecutive non-alphnum-space characters with a space
    4. remove remaining non-alphnum-space characters
    5. remove some English-language indicators

    :param series:
    :return:
    """
    return series.str.replace(r'<.*?>', '')\
        .str.lower()\                               
        .str.replace(r'[^a-z0-9 ]{2,}', ' ')\
        .str.replace(r'[^a-z0-9 ]', '')\
        .str.replace(r'\W(en|enus)\W', '')


train_test_normalized_features.info()


for column in train_test_normalized_features.columns:
    if column in ALL_TEXT_FEATURES:
        train_test_normalized_features[column] = clean_text(train_test_normalized_features[column])







# load
if 'train_test_feature_pairs.csv' in os.listdir():
    train_test_feature_pairs = reduce_mem_usage(pd.read_csv('train_test_feature_pairs.csv'))
    print(train_test_feature_pairs.info(memory_usage='deep'))

    # get the train & test indices
    train_indices, test_indices = train_test_feature_pairs.dataset.astype('object').apply(lambda x: x == 'train'),\
        train_test_feature_pairs.dataset.astype('object').apply(lambda x: x == 'test')

    # get the labels
    all_labels = train_test_feature_pairs.label
    train_labels, test_labels = all_labels[train_indices], all_labels[test_indices]

    # create feature variables
    features_regex_1, features_regex_2 = r'(' + '|'.join(COLUMNS_TO_NORMALIZE) + ')_1',\
                                         r'(' + '|'.join(COLUMNS_TO_NORMALIZE) + ')_2'


    features_1 = train_test_feature_pairs.columns[train_test_feature_pairs.columns.str.match(features_regex_1)]
    features_2 = train_test_feature_pairs.columns[train_test_feature_pairs.columns.str.match(features_regex_2)]

    feature_dtypes = train_test_feature_pairs.dtypes.astype('str')[train_test_feature_pairs.columns.str.match(features_regex_1)]

    distance_vector_features = pd.DataFrame()


    for feature_dtype, comparison_feature, feature_1, feature_2 in zip(feature_dtypes, COLUMNS_TO_NORMALIZE, features_1, features_2):
        print(feature_dtype, comparison_feature, feature_1, feature_2)

        if comparison_feature in TEXT_FEATURES and comparison_feature not in distance_vector_features.columns:
            # comparison_feature, feature_1, feature_2 = 'name', 'name_1', 'name_2'
            # comparison_feature, feature_1, feature_2 = 'description', 'description_1', 'description_2'
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

            distance_vector_features[comparison_feature] = pd.concat(distances_list, ignore_index=True)
        elif comparison_feature not in distance_vector_features.columns:
            distance_vector_features[comparison_feature] = pd.Series(train_test_feature_pairs[feature_1].astype('object')\
                                                                     == train_test_feature_pairs[feature_2].astype('object')).astype('int8')

    distance_vector_features.info(memory_usage='deep')

    train_features, test_features = distance_vector_features.loc[train_indices, ],\
        distance_vector_features.loc[test_indices, :]

