# !/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Created on Feb 10, 2019
@author: Kyle Gilde

These are the functions that are used in the 01-parse-json-to-dfs.py script

"""


import os
from urllib.parse import urlparse
import numpy as np
import pandas as pd

# functions for reading the test and train offer pairs

def read_train_test_files(file_dir, sep='#####', col_names=['offer_id_1', 'offer_id_2', 'label']):
    """
    Read all files from the director & use the file name for the category column
    :param file_dir:
    :param delimitor:
    :param category_position:
    :return:
    """
    original_wd = os.getcwd()
    os.chdir(file_dir)
    files = os.listdir()

    df_list = []
    for file in files:
        df = pd.read_csv(file, names=col_names, sep=sep, engine='python')
        df['filename'] = file
        if 'train' in file:
            df['dataset'] = 'train'
        elif 'gs_' in file:
            df['dataset'] = 'test'
        df_list.append(df)
    os.chdir(original_wd)
    return reduce_mem_usage(pd.concat(df_list, axis = 0, ignore_index=True))


def rbind_train_test_offers(train_df, test_df):
    """

    :param train_df:
    :param test_df:
    :return:
    """
    # row bind all the offer ids
    df1, df2, df3, df4 = test_df[['offer_id_1', 'filename', 'dataset']].rename(columns={'offer_id_1': 'offer_id'}),\
        test_df[['offer_id_2', 'filename', 'dataset']].rename(columns={'offer_id_2': 'offer_id'}),\
        train_df[['offer_id_1', 'filename', 'dataset']].rename(columns={'offer_id_1': 'offer_id'}),\
        train_df[['offer_id_2', 'filename', 'dataset']].rename(columns={'offer_id_2': 'offer_id'})

    # Aggregate the IDs
    train_test_offer_ids = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True, sort=False)\
        .drop_duplicates().set_index('offer_id')

    return train_test_offer_ids

# functions for parsing the massive json file

def get_cluster_ids(json_file='D:/Documents/Large-Scale Product Matching/clusters_english.json'):
    """
    Get cluster IDs for the 4 categories in the train-test sets
    :param json_file:
    :return:
    """
    if 'category_cluster_ids.csv' in os.listdir():
        category_cluster_ids = reduce_mem_usage(pd.read_csv('category_cluster_ids.csv',
                                                            index_col='cluster_id'))
    else:
        product_categories_df = reduce_mem_usage(pd.read_json(json_file, lines=True))


        category_cluster_ids = product_categories_df\
            .rename(columns={'id':'cluster_id'})\
            .set_index('cluster_id').loc[product_categories_df.category.isin(TRAIN_TEST_CATEGORIES).values, ['category']]

        print(category_cluster_ids.info(memory_usage='deep'))

        # get only the cluster ids needed for the train-test categories
        category_cluster_ids.to_csv('category_cluster_ids.csv')

        return category_cluster_ids


def merge_nan_rows(df):
    """
    Merges rows with NaNs into one
    It's passed the parse_json_column() function
    :param df: a DataFrame
    :return: one row without NaNs
    """
    try:
        s = df.apply(lambda x: x.dropna().max())
        return pd.DataFrame(s).transpose()
    except Exception as e:
        print('merge_nan_rows', e)


def parse_json_column(a_series):
    """
    Parses the remaining JSON into a DF
    :param a_series: a pandas Series
    :return: a DF
    """

    # Concatenate DFs and remove the beginning & ending brackets
    df = pd.concat([merge_nan_rows(x) for x in a_series], sort=True)\
        .apply(lambda x: x.str.replace('^\[|\]$', ''))
    # clean column names
    df.columns = df.columns.str.strip('/')

    return df


def coalesce_gtin_columns(df):
    """
    Since a product can have only one gtin,
    this function coalesces these columns to one column
    :param df:
    :return:
    """
    # select the gtin columns
    gtin_df = df.filter(regex='gtin')
    gtin = gtin_df.iloc[:, 0]
    if len(gtin_df.columns) > 1:
        # start the loop on the 2nd column
        for col in gtin_df.columns[1:]:
            gtin = gtin.mask(pd.isnull, gtin_df[col])
    df['gtin'] = gtin
    df.drop(gtin_df.columns, axis=1, inplace=True)
    return df


def parse_domain(url):
    """

    :param url:
    :return:
    """
    return urlparse(url).netloc


def parse_price_columns(df, price_column_names):
    """

    :param price_series:
    :return:
    """
    price_columns = df.columns[df.columns.isin(price_column_names)]

    for price_column in price_columns:
        df[price_column] = df[price_column]\
            .str.replace(r'[a-zA-Z",]+', '')\
            .str.strip()\
            .str.replace(r' (\d\d)$', r'.\1')\
            .str.replace(r'\s', '')\
            .apply(lambda x: pd.to_numeric(x, errors='coerce', downcast='integer'))

    return df


def coalesce_parent_child_columns(df):
    """

    :param df:
    :return:
    """
    parent_columns = df.columns[df.columns.astype(str).str.startswith('parent_')]
    nonparent_columns = df.columns[~df.columns.isin(parent_columns)]

    child_parent_pairs = []
    for parent_column in parent_columns:
        for nonparent_column in nonparent_columns:
            if parent_column.endswith('_' + nonparent_column):
                child_parent_pairs.append((parent_column, nonparent_column))

    for child_parent_pair in child_parent_pairs:
        parent, child = child_parent_pair
        df[child] = df[child].combine_first(df[parent])
        df.drop(parent, axis=1, inplace=True)

    return df


def create_file_categories(df):
    """
    Creates a df column for the test and train category

    :param df: the df containing a column called filename
    :return: the df with a column called file_category
    """
    if 'filename' in df.columns:

        file_categories = ['computers', 'cameras', 'watches', 'shoes']

        for file_category in file_categories:
            df.loc[df['filename'].str.contains(file_category), 'file_category'] = file_category

        return df
    else:
        print('The df does not have a column called filename')


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
