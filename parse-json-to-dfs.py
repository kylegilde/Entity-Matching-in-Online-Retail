# !/usr/bin/env/python365
"""
Created on Feb 10, 2019
@author: Kyle Gilde
"""
import os
import gc
from datetime import datetime

from urllib.parse import urlparse
import re
import string

import sys
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from pandas.api.types import is_numeric_dtype, is_string_dtype
import matplotlib.pyplot as plt
import seaborn as sns


import json
import psutil
import multiprocessing as mp

TRAIN_TEST_CATEGORIES = ['Computers_and_Accessories', 'Camera_and_Photo', 'Shoes', 'Jewelry']
PRICE_COLUMN_NAMES = ['price','parent_price']
os.chdir('D:/Documents/Large-Scale Product Matching/')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

def reduce_mem_usage(df, n_unique_object_threshold=0.30):
    """
    source: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    :param df:
    :return:
    """
    print("------------------------------------")
    start_mem_usg = df.memory_usage().sum() / 1024**2
    print("Starting memory usage is %s MB" % "{0:}".format(start_mem_usg))
    # record the dtype changes
    dtype_df = pd.DataFrame(df.dtypes.astype('str'), columns=['original'])

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            # make variables for max, min
            mx, mn = df[col].max(), df[col].min()
            # If no NaNs, proceed to reduce the int
            if np.isfinite(df[col]).all():
                # test if column can be converted to an integer
                as_int = df[col].astype(np.int64)
                delta = (df[col] - as_int).sum()

                # Make Integer/unsigned Integer datatypes
                if delta == 0:
                    if mn >= 0:
                        if mx < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif mx < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif mx < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                        else:
                            df[col] = df[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)

                # Make float datatypes 32 bit
                else:
                    df[col] = df[col].astype(np.float32)

                # Print new column type
                # print("dtype after: ", df[col].dtype)
        elif is_string_dtype(df[col]):
            if df[col].astype(str).nunique() / len(df) < n_unique_object_threshold:
                df[col] = df[col].astype('category')

    # Print final result
    dtype_df['new'] = df.dtypes.astype('str')
    dtype_changes = dtype_df.original != dtype_df.new

    if dtype_changes.sum():
        print(dtype_df.loc[dtype_changes])
        new_mem_usg = df.memory_usage().sum() / 1024**2
        print("Ending memory usage is %s MB" % "{0:}".format(new_mem_usg))
        print("Reduced by", int(100 * (1 - new_mem_usg / start_mem_usg)), "%")

    else:
        print('No reductions possible')

    print("------------------------------------")
    return df


def read_train_test_files(file_dir, delimitor='_', category_position=0,
                          col_names=['offer_id_1', 'offer_id_2', 'label']):
    """
    Read all files from the director & use the file name for the category column
    :param file_dir:
    :param delimitor:
    :param category_position:
    :return:
    """
    os.chdir(file_dir)
    files = os.listdir()

    df_list = []
    for file in files:
        df = pd.read_csv(file, names=col_names, sep = '#####', engine='python')
        df['category_file'] = re.split(delimitor, file)[category_position]
        df_list.append(df)

    return pd.concat(df_list, axis = 0, ignore_index = True)


def rbind_train_test_offers(train_df, test_df):
    """

    :param train_df:
    :param test_df:
    :return:
    """
    # row bind all the offer ids
    df1, df2, df3, df4 = test_df[['offer_id_1', 'category_file']].rename(columns={'offer_id_1': 'offer_id'}),\
        test_df[['offer_id_2', 'category_file']].rename(columns={'offer_id_2': 'offer_id'}),\
        train_df[['offer_id_1', 'category_file']].rename(columns={'offer_id_1': 'offer_id'}),\
        train_df[['offer_id_2', 'category_file']].rename(columns={'offer_id_2': 'offer_id'})

    # Aggregate the IDs
    train_test_offer_ids = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True, sort=False)\
        .drop_duplicates().set_index('offer_id')

    return train_test_offer_ids


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


def parse_price(df, price_column_names=PRICE_COLUMN_NAMES):
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

def count_nulls_by_column(df):
    """

    :param df:
    :return:
    """
    print(np.round(df.isnull().sum() / len(df) * 100), 1)

##################################
#### Load Train and Test Data ####
##################################
# Load Train Data
train_df = reduce_mem_usage(read_train_test_files('D:/Documents/Large-Scale Product Matching/training-data/'))
plt.gcf().clear()
train_df.category_file.value_counts().plot.bar()

# Load Test Data
test_df = reduce_mem_usage(read_train_test_files('D:/Documents/Large-Scale Product Matching/test-data/',
                                category_position=1))
plt.gcf().clear()

test_df.category_file.value_counts().plot.bar()

train_test_df = pd.concat([train_df, test_df], axis=0)

train_test_offer_ids = rbind_train_test_offers(train_df, test_df)


#######################################
#### Clean & Tidy the Offers data ####
#######################################
# if file exists read it, otherwise create it
if 'category_offers_df.csv' in os.listdir():
    category_offers_df = reduce_mem_usage(pd.read_csv('category_offers_df.csv',
                                                      index_col='offer_id')).drop('Unnamed: 0', axis=1)
    print(category_offers_df.info(memory_usage='deep'))
else:

    start = datetime.now()
    # Read or Create the Cluster ID-Category Mappings
    category_cluster_ids = get_cluster_ids()

    # get offers data for train-test categories
    offers_reader = pd.read_json('offers_consWgs_english.json',
                                 orient='records',
                                 chunksize=1e6,
                                 lines=True)

    columns_with_json = ['identifiers', 'schema_org_properties', 'parent_schema_org_properties']
    columns_to_drop = ['parent_NodeID', 'relationToParent',] + columns_with_json
    more_cols_to_drop = ['identifier', 'productID', 'sku', 'parent_availability', 'availability']


    offers_df_list = []
    for i, chunk in enumerate(offers_reader):
        print(i)
        # chunk = next(offers_reader)
        # inner join on the cluster ids for the 4 train-test categories
        new_chunk = reduce_mem_usage(chunk)\
                       .set_index('cluster_id')\
                       .join(category_cluster_ids, how='inner')

        # clean column names
        new_chunk.columns = new_chunk.columns.str.replace('.', '_')
        # create the unique offer_id
        new_chunk['offer_id'] = new_chunk.nodeID.str.cat(new_chunk.url, sep=' ')
        # parse the website domain into column
        new_chunk['domain'] = new_chunk.url.apply(parse_domain)
        # set offer_id to index and drop its components
        new_chunk = new_chunk.reset_index()\
            .set_index('offer_id')\
            .drop(['nodeID', 'url'], axis=1)

        parsed_df_list = []
        json_columns_df = new_chunk[columns_with_json]
        for column in columns_with_json:
            # column = columns_with_json[2]
            print(column)
            df = parse_json_column(json_columns_df[column].apply(json_normalize))\
                .apply(lambda x: x.str.strip() if x.dtype == "object" else x)\
                .replace('null', np.nan)
            print(df.columns)

            df.index = new_chunk.index
            # coalesce the gtins
            if column == 'identifiers':
                df = coalesce_gtin_columns(df)
            elif column == 'parent_schema_org_properties':
                df.columns = 'parent_' + df.columns
            # parse the price columns
            if np.sum(df.columns.isin(price_column_names)):
                df = parse_price(df, price_column_names)

            parsed_df_list.append(df)

        print(datetime.now() - start)
        # Drop the 3 parsed columns
        new_chunk.drop(columns_to_drop, axis=1, inplace=True)
        # Concatenate the chunk to the 3 parsed columns & add it to the df list
        parsed_df_list.append(new_chunk)
        new_chunk = reduce_mem_usage(pd.concat(parsed_df_list, axis=1, sort=False))
        offers_df_list.append(new_chunk)

    # save the output
    print('Saving as CSV...')
    category_offers_df = reduce_mem_usage(pd.concat(offers_df_list, axis=0).drop(more_cols_to_drop, axis=1))
    category_offers_df.to_csv('category_offers_df.csv', index_label='offer_id')

    print(category_offers_df.describe(include='all'))
    print(category_offers_df.info(memory_usage=True))
    print(category_offers_df.isnull().sum() / len(category_offers_df) * 100)


# create df for only the offers in the train and test sets
train_test_offers_df = category_offers_df.join(train_test_offer_ids, how='inner')

print(train_test_offers_df.describe(include='all'))
print(train_test_offers_df.info(memory_usage=True))



train_test_df_details = train_test_df.set_index('offer_id_1')\
    .join(train_test_offers_df.add_suffix('_1'), how='left')\
    .reset_index()\
    .set_index('offer_id_2')\
    .join(train_test_offers_df.add_suffix('_2'), how='left')\
    .sort_index(axis=1)\
    .set_index('label')\
    .sort_index(axis=0)\
    .reset_index()\
    .dropna(axis=1)#.drop(columns=['availability_1', 'availability_2'])

print(train_test_df_details.describe(include='all'))
print(train_test_df_details.info(memory_usage=True))
count_nulls_by_column(train_test_df_details)





    print(category_offers_df.info(memory_usage=True))





    category_offers_df.head()

    np.sum(category_offers_df.index.duplicated())
    len(category_offers_df.index.unique())

    print(df['brand'].dtype)
    isinstance(df['brand'], object)
#######################################
# Parse the remaining JSON in 3 columns
#######################################
# Read in Chunked CSV
# start = datetime.now()

#
# category_offers_df = pd.read_csv('category_offers_df.csv', chunksize=5e5)
# type(category_offers_df)
#
# columns_with_json = ['identifiers', 'schema.org_properties', 'parent_schema.org_properties']
# columns_to_drop = ['parent_NodeID', 'relationToParent'] + columns_with_json
#
# new_df_list = []
# for i, chunk in enumerate(category_offers_df):
#     # Parse each column
#     print(i)
#     #chunk = next(category_offers_df)
#     parsed_df_list = []
#     # json_columns_df = clean_json_columns(chunk.loc[:, columns_with_json])
#     json_columns_df = clean_json_columns(chunk[columns_with_json])
#     for column in columns_with_json:
#         print(column)
#         df = parse_json_column(json_columns_df[column], column_prefix=column)\
#             .reset_index(drop=True)
#         # coalesce the gtins
#         if column == 'identifiers':
#             df = coalesce_gtin_columns(df)
#         parsed_df_list.append(df)
#         #sys.getsizeof(parsed_df_list)
#     # Drop the 3 parsed columns
#     chunk.drop(columns_to_drop, axis=1, inplace=True)
#     # Concatenate the chunk to the 3 parsed columns & add it to the df list
#     parsed_df_list.append(chunk)
#     new_df = pd.concat(parsed_df_list, axis=1, sort=False)
#     new_df_list.append(new_df)
#
#
# # my_df = reduce_mem_usage(pd.concat(new_df_list, axis=0))
# print('Saving as CSV...')
# reduce_mem_usage(pd.concat(new_df_list, axis=0)).to_csv('category_offers_df_v2.csv')
# print(datetime.now() - start)


# my_df[['schema.org_properties_price', 'parent_schema.org_properties_price']]
# new_price = my_df['parent_schema.org_properties_price'].str.replace('(usd|,|gbp)', '')\
#     .str.strip()\
#     .str.replace(' ', '.')\
#     .astype('float')
#
# new_price.value_counts()


# gc.collect()
# print(psutil.virtual_memory())
# parsed_df_list = []
#
#
# gc.collect()
#
# print(datetime.now() - start)
#
#
# parsed_df.info(memory_usage='deep')
# print('Saving as CSV...')
# parsed_df.to_csv('parsed_df.csv')







# Load the Offers
# os.chdir('D:/Documents/Large-Scale Product Matching/')
# offers_df = pd.read_json('offers_consWgs_english.json',
#                          orient='records',
#                          chunksize=1e4,
#                          lines=True)
#
# one_chunk = next(offers_df)
#
#
#
# one_chunk.columns
# one_chunk.info()
# json_normalize(list(one_chunk['identifiers'][:1000]))
#
# json_normalize(one_chunk['identifiers'][:10])
#
# a = one_chunk['identifiers'][:10].apply(lambda x: json_normalize(x))
#
#
# json_normalize(one_chunk['identifiers'][:10], meta=['/mpn', '/sku', '/productID', '/gtin8', '/gtin13'])
# one_chunk.head()
#
# # read json using chunks
#
# df2 = pd.concat([pd.DataFrame(json_normalize(x)) for x in one_chunk['schema.org_properties']], ignore_index=True)
# df2.columns = df2.columns.str.strip('/')
#
# new_df = pd.concat([pd.DataFrame(flatten(x)) for x in one_chunk['identifiers']], ignore_index=True)
#
# flatten(dict(one_chunk['identifiers']))
#
# new_df.columns = new_df.columns.str.strip('/')
#
# type(new_df['mpn'])



# offers_file_path = 'D:/Documents/Large-Scale Product Matching/samples/sample_offersenglish.json'
# with open('offers_consWgs_english.json', encoding="utf8") as offers_file:
#     offers_json = [flatten(json.loads(record)) for record in offers_file]
# offers_file.close()
# gc.collect()
# print('file closed')
#
# with open('flattened_offers.json', 'w', encoding="utf8") as flattened_offers:
#     json.dump(offers_json, flattened_offers)
# https://stackoverflow.com/a/46482086/4463701
# flattened_offers.close()
# # del(offers_json)
# os.chdir('D:/Documents/Large-Scale Product Matching/')
# with open('flattened_offers.json', encoding="utf8") as flattened_offers:
#     offers_json = [dict(json.loads(record)) for record in flattened_offers]
#     pd.DataFrame(offers_json).to_csv('offers_df.csv', index=False)


# offers_df.to_csv('D:/Documents/Large-Scale Product Matching/offers_df.csv', index=False)
# offers_df = pd.concat(offers_json, ignore_index=True)
#offers_df = pd.DataFrame([flatten(record) for record in offers_json])
#
# offers_df.info()
# # offers_df.head()
# def clean_json_columns(df, removal_class=CHARS_TO_REMOVE_CLASS):
#     """
#
#     :param df:
#     :param removal_class:
#     :return:
#     """
#     for col in df.columns:
#         try:
#             df.loc[:, col] = df.loc[:, col].str.replace(removal_class, '')\
#                 .str.replace("'", '"')\
#                 .apply(load_and_normalize_json)
#         except Exception as e:
#             print('clean_json_columns:', e)
#
#     return df
# def load_and_normalize_json(a_string):
#     try:
#         return json_normalize(json.loads(a_string))
#     except Exception as e:
#         print(e)
#         print(a_string)
