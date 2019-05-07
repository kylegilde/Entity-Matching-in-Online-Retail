# !/usr/bin/env/python365
"""
Created on Apr 27, 2019
@author: Kyle Gilde

This script takes 2 input files:

    1. 'train_test_stemmed_features.csv' created from 02-normalized-features.py
    2. 'train_test_df.csv' created from 01-parse-json-to-dfs.py

For each of the 4 dataset categories, this script compares each of the corresponding attribute values for each pair of offers.

    1. For the 7 short-to-medium length text features, it calculates the scaled Levenshtein similarity score.
    2. For the long description feature, it encodes the text as bag-of-words TF-IDF approach for unigrams, bigrams & trigrams.
        Truncated SVD is used to select only the components that explain 99.9% of the variances, and then the cosine similarity
        is calculated for the 2 vectors.
    3. For the price, the absolute percent diffence is calculated
    4. For the strongly-typed offer category, a 0 or 1 indicates whether they are an exact match.

It outputs symbolic_similarity_features.csv,
 which is a dataframe that contains the similarity vectors of the offer pairs
  for both the test and training sets.

"""

import os
import gc

from datetime import datetime
import nltk
import numpy as np
import pandas as pd

from json_parsing_functions import reduce_mem_usage
from utility_functions import *

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



start_time = datetime.now()

# set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 250)

# initialize constants
DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
DATA_DIRECTORY = '//files/share/goods/OI Team'
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

# the description column must be last in the list
ALL_FEATURES = ALL_SHORT_TEXT_FEATURES + STRONGLY_TYPED_FEATURES + NUMERIC_FEATURE + LONG_TEXT_FEATURES

OFFER_PAIR_COLUMNS = ['offer_id_1', 'offer_id_2', 'filename', 'dataset']

# set display options
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 0)

# load files
assert 'train_test_stemmed_features.csv' in os.listdir() and 'train_test_df.csv' in os.listdir(), 'An input file is missing'

train_test_stemmed_features = reduce_mem_usage(pd.read_csv('train_test_stemmed_features.csv'))\
    .set_index('offer_id')

train_test_df = reduce_mem_usage(pd.read_csv('train_test_df.csv'))


file_categories = train_test_df.file_category.unique()
category_df_list = []
start_time = datetime.now()

for the_category in file_categories:

    print('the_category:', the_category)
    # the_category = 'shoes'

    cat_symbolic_similarity_features = train_test_df[train_test_df.file_category == the_category].copy()
    unique_offer_ids = pd.concat([cat_symbolic_similarity_features.offer_id_1.astype('object'),
                                  cat_symbolic_similarity_features.offer_id_2.astype('object')])\
        .unique()

    cat_symbolic_similarity_features.set_index(OFFER_PAIR_COLUMNS, inplace=True)

    for column in ALL_FEATURES:
        print('column:', column)
        get_duration_hours(start_time)

        # put the left and right side feature into a df
        both_features =\
            cat_symbolic_similarity_features.reset_index()[OFFER_PAIR_COLUMNS]\
            .set_index(OFFER_PAIR_COLUMNS[0], drop=False)\
            .join(train_test_stemmed_features[[column]].add_suffix('_1'), how='inner')\
            .set_index(OFFER_PAIR_COLUMNS[1], drop=False)\
            .join(train_test_stemmed_features[[column]].add_suffix('_2'), how='inner')\
            .set_index(OFFER_PAIR_COLUMNS)

        if column in ALL_SHORT_TEXT_FEATURES:

            # scaled Levenshtein similarity score
            cat_symbolic_similarity_features[column] = both_features.apply(levenshtein_similarity, axis=1)
            cat_symbolic_similarity_features = reduce_mem_usage(cat_symbolic_similarity_features)

        elif column in STRONGLY_TYPED_FEATURES:

            # exact match
            cat_symbolic_similarity_features[column] = pd.Series(both_features.iloc[:, 0]\
                                                             == both_features.iloc[:, 1]).astype('int8')
            cat_symbolic_similarity_features = reduce_mem_usage(cat_symbolic_similarity_features)

        elif column in NUMERIC_FEATURE:

            # absolute percent difference
            cat_symbolic_similarity_features[column] = \
                np.nan_to_num(np.absolute(both_features.iloc[:, 0] - both_features.iloc[:, 1]) / \
                              np.maximum(both_features.iloc[:, 0], both_features.iloc[:, 1]))
            cat_symbolic_similarity_features = reduce_mem_usage(cat_symbolic_similarity_features)

        elif column in LONG_TEXT_FEATURES:
            # column = 'description'

            # bag-of-words TF-IDF with Truncated SVD and cosine similarity
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))

            # create a document-term matrix from the unique column values
            unique_column_values = train_test_stemmed_features.loc[unique_offer_ids][[column]].fillna('')

            dtm = vectorizer.fit_transform(unique_column_values[column])
            print('dtm dimensions:', dtm.shape)

            get_duration_hours(start_time)
            print('Use Truncated SVD to select a smaller number of important features')
            svd_model = TruncatedSVD(n_components=MAX_SVD_COMPONENTS).fit(dtm)
            print('n_components:', len(svd_model.explained_variance_ratio_))
            print(svd_model.explained_variance_ratio_.sum(), 'variance explained')

            n_features = sum(svd_model.explained_variance_ratio_.cumsum() <= VARIANCE_EXPLAINED)
            print(n_features, "features explain this much of the variance:", VARIANCE_EXPLAINED)

            print('fit the svd model and convert to df')
            dtm_svd = reduce_mem_usage(pd.DataFrame(svd_model.transform(dtm)[:, :n_features],
                                                    index=unique_column_values.index))

            print('post-SVD DTM dimensions:', dtm_svd.shape)
            print(dtm_svd.info(memory_usage='deep'))

            del dtm, svd_model, unique_column_values
            gc.collect()

            get_duration_hours(start_time)
            print("Let's create a df to hold both sides of the DTM")

            both_sides_dtm_svd =\
                reduce_mem_usage(
                    both_features
                        .reset_index()
                        .drop(['description_1', 'description_2'], axis=1)
                        .set_index('offer_id_1', drop=False)
                        .join(dtm_svd.add_suffix('_1'), how='inner')
                        .set_index('offer_id_2', drop=False)
                        .join(dtm_svd.add_suffix('_2'), how='inner')
                        .set_index(OFFER_PAIR_COLUMNS)
                )

            print(both_sides_dtm_svd.info())

            del both_features
            gc.collect()

            get_duration_hours(start_time)
            print("Let's calculate the cosine similarity.")
            assert both_sides_dtm_svd.shape[1] == 2 * n_features, "Something is wrong. Your df row length is not 2 x n_features"
            cat_symbolic_similarity_features[column] = both_sides_dtm_svd.apply(elementwise_cosine_similarity,
                                                                            n_features=n_features,
                                                                            axis=1)
            del both_sides_dtm_svd
            gc.collect()

            # append category DF
            category_df_list.append(cat_symbolic_similarity_features)

    print('Combine all the categories')
    symbolic_similarity_features =\
        reduce_mem_usage(pd.concat(category_df_list, axis=0))\
        .reset_index()

    print('Summary stats')
    print(symbolic_similarity_features.columns)
    print(symbolic_similarity_features.shape)
    print(symbolic_similarity_features.describe())
    print(symbolic_similarity_features.info())

    print("symbolic_similarity_features saved")
    symbolic_similarity_features.to_csv('symbolic_similarity_features.csv', index=False)

    get_duration_hours(start_time)
