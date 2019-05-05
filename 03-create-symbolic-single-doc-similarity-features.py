# !/usr/bin/env/python365
"""
Created on Apr 27, 2019
@author: Kyle Gilde

This script takes the outputs of create-symbolic-features.py
and parse-json-to-dfs.py.

It outputs a df that contains the similarity vectors of the offer pairs
in the test and training sets. The df is saved as cat_symbolic_similarity_features.csv

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
MAX_SVD_COMPONENTS = 3000
VARIANCE_EXPLAINED = 0.999
N_ROWS_PER_ITERATION = 2000

# the description column must be last in the list
ALL_FEATURES = ['name', 'description', 'brand', 'manufacturer', 'gtin', 'mpn', 'sku', 'identifier', 'price'] #'category'

OFFER_PAIR_COLUMNS = ['offer_id_1', 'offer_id_2', 'filename', 'dataset']

# set display options
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 0)

# load files
assert 'train_test_stemmed_features.csv' in os.listdir() and 'train_test_df.csv' in os.listdir(), 'An input file is missing'

train_test_stemmed_features = pd.read_csv('train_test_stemmed_features.csv')\
    .set_index('offer_id')

print(train_test_stemmed_features.reset_index().head(100))

train_test_df = reduce_mem_usage(pd.read_csv('train_test_df.csv'))


file_categories = train_test_df.file_category.unique()
category_df_list = []
start_time = datetime.now()

# for the_category in file_categories:
#
#     print('the_category:', the_category)
#     # the_category = 'shoes'
#
#     cat_symbolic_similarity_features = train_test_df[train_test_df.file_category == the_category].copy()
#     unique_offer_ids = pd.concat([cat_symbolic_similarity_features.offer_id_1.astype('object'),
#                                   cat_symbolic_similarity_features.offer_id_2.astype('object')])\
#         .unique()

cat_symbolic_similarity_features = train_test_df.copy()
unique_offer_ids = pd.concat([cat_symbolic_similarity_features.offer_id_1.astype('object'),
                              cat_symbolic_similarity_features.offer_id_2.astype('object')])\
    .unique()

cat_symbolic_similarity_features.set_index(OFFER_PAIR_COLUMNS, inplace=True)


train_test_stemmed_features['price'] = train_test_stemmed_features.price.astype('str')
unique_column_values = train_test_stemmed_features.apply(lambda x: ' '.join(x.dropna()), axis=1)

print(unique_column_values.reset_index().head(100))

vectorizer = TfidfVectorizer(ngram_range=(1, 3))

dtm = vectorizer.fit_transform(unique_column_values)
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
        cat_symbolic_similarity_features\
            .reset_index()
            .set_index('offer_id_1', drop=False)
            .join(dtm_svd.add_suffix('_1'), how='inner')
            .set_index('offer_id_2', drop=False)
            .join(dtm_svd.add_suffix('_2'), how='inner')
            .set_index(OFFER_PAIR_COLUMNS)
    )

print(both_sides_dtm_svd.info())

# del both_features
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
