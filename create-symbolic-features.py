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

import nltk
from nltk.corpus import stopwords
sb_stemmer = nltk.stem.SnowballStemmer('english')


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


def stem_and_remove_stopwords(doc, stemmer):
    """

    :param doc:
    :param stemmer:
    :return:
    """
    if pd.isnull(doc):
        return doc
    else:
        stemmed_tokens = [stemmer.stem(token) for token in nltk.word_tokenize(doc)\
                          if token not in stopwords.words('english')]

        return ' '.join(stemmed_tokens)


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

ALL_FEATURES = ALL_TEXT_FEATURES + STRONGLY_TYPED_FEATURES + NUMERIC_FEATURE

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


train_test_normalized_features = train_test_offer_features[ALL_FEATURES]

# loop through text features and clean them
for column in train_test_normalized_features.columns:
    if column in ALL_TEXT_FEATURES:
        train_test_normalized_features[column] = clean_text(train_test_normalized_features[column])

# save to file
train_test_normalized_features.reset_index().to_csv('train_test_normalized_features.csv', index=False)

train_test_stemmed_features = train_test_normalized_features.copy()

for column in train_test_stemmed_features.columns:
    if column in TEXT_FEATURES:
        train_test_stemmed_features[column] = train_test_normalized_features[column]\
            .apply(lambda x: stem_and_remove_stopwords(x, stemmer=sb_stemmer))

# save to file
train_test_stemmed_features.reset_index().to_csv('train_test_stemmed_features.csv', index=False)

train_test_normalized_features.description[:10].reset_index(drop=True)
train_test_stemmed_features.description[:10].reset_index(drop=True)
