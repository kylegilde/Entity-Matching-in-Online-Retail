# !/usr/bin/env/python365
"""
Created on Apr 27, 2019
@author: Kyle Gilde

This script takes the outputs of create-symbolic-features.py
and parse-json-to-dfs.py.

It outputs a df that contains the similarity vectors of the offer pairs
in the test and training sets. The df is saved as symbolic_single_doc_similarity_features.csv

"""

import os
import gc

from datetime import datetime
import nltk
import numpy as np
import pandas as pd

from utility_functions import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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
MAX_SVD_COMPONENTS = 2000
VARIANCE_EXPLAINED_MAX = 0.999
N_ROWS_PER_ITERATION = 2000

# the description column must be last in the list
ALL_FEATURES = ['name', 'description', 'brand', 'manufacturer', 'gtin', 'mpn', 'sku', 'identifier', 'price'] #'category'

OFFER_PAIR_COLUMNS = ['offer_id_1', 'offer_id_2', 'filename', 'dataset', 'label', 'file_category']

# load files
assert 'train_test_stemmed_features.csv' in os.listdir() and 'train_test_df.csv' in os.listdir(), 'An input file is missing'

print('load the stemmed features')
train_test_stemmed_features = pd.read_csv('train_test_stemmed_features.csv')\
    .set_index('offer_id')
print(train_test_stemmed_features.info())

print('load the offer pairs')
train_test_df = reduce_mem_usage(pd.read_csv('train_test_df.csv')\
                                .set_index(OFFER_PAIR_COLUMNS))
print(train_test_df.info())
assert len(train_test_df.columns) == 0, 'The DF contains extra column(s)'

start_time = datetime.now()

print("Let's concatenate all features into single documents")

train_test_stemmed_features['price'] = train_test_stemmed_features.price.astype('str')
unique_column_values =\
    train_test_stemmed_features[ALL_FEATURES]\
        .apply(lambda x: ' '.join(x.dropna()), axis=1)\
        .str.replace('\Wnan$', '')

unique_column_value_indices = unique_column_values.index

print("Let's take a look at the concatenation")
print(unique_column_values.reset_index().head())

# reclaim some memory
del train_test_stemmed_features
gc.collect()

print("Let's create a DTM using TF-IDF")
vectorizer = TfidfVectorizer(ngram_range=(1, 3))

dtm = vectorizer.fit_transform(unique_column_values)
print('dtm dimensions:', dtm.shape)

# reclaim some memory
del unique_column_values
gc.collect()

get_duration_hours(start_time)
print('Use Truncated SVD to select a smaller number of important features')
svd_model = TruncatedSVD(n_components=MAX_SVD_COMPONENTS).fit(dtm)

print('n_components:', len(svd_model.explained_variance_ratio_))
variance_explained = svd_model.explained_variance_ratio_.sum()
print(variance_explained, 'variance explained')

n_features = sum(svd_model.explained_variance_ratio_.cumsum() <= VARIANCE_EXPLAINED_MAX)
print(n_features, "features explain this much of the variance:", variance_explained)

print('fit the svd model and convert to df')
dtm_svd = reduce_mem_usage(pd.DataFrame(svd_model.transform(dtm)[:, :n_features],
                                        index=unique_column_value_indices))

print('post-SVD DTM dimensions:', dtm_svd.shape)
print(dtm_svd.info(memory_usage='deep'))

# reclaim some memory
del dtm, svd_model
gc.collect()

get_duration_hours(start_time)
train_test_df.info()
print("Let's create a df to hold both sides of the DTM")
symbolic_single_doc_similarity_features =\
    reduce_mem_usage(
        train_test_df\
            .reset_index()
            .set_index('offer_id_1', drop=False)
            .join(dtm_svd.add_suffix('_1'), how='inner')
            .set_index('offer_id_2', drop=False)
            .join(dtm_svd.add_suffix('_2'), how='inner')
            .set_index(OFFER_PAIR_COLUMNS)
    )
print(symbolic_single_doc_similarity_features.info())

assert symbolic_single_doc_similarity_features.shape[1] == 2 * n_features, "Something is wrong. Your df row length is not 2 x n_features"

print('Calculate the absolute difference between each offer pair')
symbolic_single_doc_similarity_features_df =\
    (symbolic_single_doc_similarity_features.iloc[:, :n_features]\
    .sub(symbolic_single_doc_similarity_features.iloc[:, n_features:].values))\
    .abs()

del symbolic_single_doc_similarity_features
gc.collect()

print('Summary stats')
print(symbolic_single_doc_similarity_features_df.shape)
print(symbolic_single_doc_similarity_features_df.info())
print(symbolic_single_doc_similarity_features_df.columns)
print(symbolic_single_doc_similarity_features_df.describe())

get_duration_hours(start_time)
print("symbolic_similarity_features saved")
symbolic_single_doc_similarity_features_df.to_csv('symbolic_single_doc_similarity_features.csv')

get_duration_hours(start_time)


# file_categories = train_test_df.file_category.unique()
# category_df_list = []

# for the_category in file_categories:
#
#     print('the_category:', the_category)
#     # the_category = 'shoes'
#
#     symbolic_single_doc_similarity_features = train_test_df[train_test_df.file_category == the_category].copy()
#     unique_offer_ids = pd.concat([symbolic_single_doc_similarity_features.offer_id_1.astype('object'),
#                                   symbolic_single_doc_similarity_features.offer_id_2.astype('object')])\
#         .unique()

# append category DF
# category_df_list.append(symbolic_single_doc_similarity_features)
#
# print('Combine all the categories')
# symbolic_similarity_features =\
#     reduce_mem_usage(pd.concat(category_df_list, axis=0))\
#     .reset_index()
# print("Let's calculate the cosine similarity.")
# assert both_sides_dtm_svd.shape[1] == 2 * n_features, "Something is wrong. Your df row length is not 2 x n_features"
# symbolic_single_doc_similarity_features[column] = both_sides_dtm_svd.apply(elementwise_cosine_similarity,
#                                                                 n_features=n_features,
#                                                                 axis=1)