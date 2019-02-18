# !/usr/bin/env/python365
"""
Created on Feb 10, 2019
@author: Kyle Gilde
"""
import os
import re
import gc
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.io.json import json_normalize
import json
import psutil
import multiprocessing as mp

mp.cpu_count()
TRAIN_TEST_CATEGORIES = ['Computers_and_Accessories', 'Camera_and_Photo', 'Shoes']
os.chdir('D:/Documents/Large-Scale Product Matching/')
def reduce_mem_usage(df, n_unique_object_threshold=0.5):
    """
    source: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    :param df:
    :return:
    """
    start_mem_usg = df.memory_usage().sum() / 1024**2
    print("Starting memory usage is:", int(start_mem_usg)," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    original_dtypes = df.dtypes

    for col in df.columns:
        if not isinstance(df[col].dtype, object):  # Exclude strings

            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ", df[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx, mn = df[col].max(), df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1,inplace=True)

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True


            # Make Integer/unsigned Integer datatypes
            if IsInt:
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
            # print("******************************")
        elif isinstance(df[col].dtype, object) and df[col].astype(str).nunique() / len(df) < n_unique_object_threshold:
            df[col] = df[col].astype('category')

    # Print final result
    new_dtypes = df.dtypes
    dtype_changes = original_dtypes != new_dtypes

    if dtype_changes.sum():
        mem_usg = df.memory_usage().sum() / 1024**2
        print("Ending memory usage is: ", int(mem_usg)," MB")
        print("Reduced by", int(100 * (1 - mem_usg / start_mem_usg)), "%")

        dtype_changes_df = pd.concat([df.columns, original_dtypes, new_dtypes], axis=1)
        # print(dtype_changes_df.filter[dtype_changes])
        if NAlist:
            print('columns with NAs:', NAlist)
    else:
        print('No reductions possible')
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
        df['category'] = re.split(delimitor, file)[category_position]
        df_list.append(df)

    return pd.concat(df_list, axis = 0, ignore_index = True)

def merge_nan_rows(df):
    """
    Merges rows with NaNs into one
    :param df: a DataFrame
    :return: one row without NaNs
    """
    s = df.apply(lambda x: x.dropna().max())
    return pd.DataFrame(s).transpose()

def parse_json_column(s, column_prefix=None):
    """
    Parses the remaining JSON into a DF
    :param s: a pandas Series
    :return: a DF
    """
    # Normalize JSON
    s2 = s.str.replace('"', '')\
        .str.replace("'", '"')\
        .apply(json.loads)\
        .apply(lambda x: json_normalize(x))
    # Concatenate DFs and remove brackets
    df = pd.concat([merge_nan_rows(x) for x in s2], sort=True)\
        .apply(lambda x: x.str.replace('^\[|\]$', ''))
    # clean column names
    df.columns = df.columns.str.strip('/')

    if column_prefix:
        df.columns = column_prefix + '_' + df.columns

    return reduce_mem_usage(df)

# product_categoreies_df.head()
# product_categoreies_df.info()
# product_categoreies_df.select_dtypes(include=['object']).describe()
# plt.gcf().clear()
# product_categoreies_df.category.value_counts().plot.bar()
# product_categoreies_df.id

# Load Train Data
train_df = read_train_test_files('D:/Documents/Large-Scale Product Matching/training-data/')
plt.gcf().clear()
train_df.category.value_counts().plot.bar()

# Load Test Data
test_df = read_train_test_files('D:/Documents/Large-Scale Product Matching/test-data/',
                                category_position=1)
plt.gcf().clear()
test_df.category.value_counts().plot.bar()

# Aggregate the IDs
train_test_offer_ids = pd.concat([test_df.offer_id_1, test_df.offer_id_2, train_df.offer_id_1, train_df.offer_id_2], axis=0,
                           ignore_index=True)

# del test_df
# # del train_df
gc.collect()
print(psutil.virtual_memory())

# Read or Create the Product-Category Mappings (category_cluster_ids)
try:
    os.chdir('D:/Documents/Large-Scale Product Matching/')
    category_cluster_ids = reduce_mem_usage(pd.read_csv('category_cluster_ids.csv',
                                                        index_col='cluster_id'))
    print(category_cluster_ids.info(memory_usage='deep'))
except Exception as e:
    print(e)
else:
    product_categories_df = reduce_mem_usage(pd.read_json('D:/Documents/Large-Scale Product Matching/clusters_english.json',
                                          lines=True))

    print(product_categories_df.info(memory_usage='deep'))
    print(product_categories_df.info(memory_usage='deep'))
    # product_categories_df['category']= product_categories_df.category.astype('category')
    product_categories_df = product_categories_df.rename(columns={'id':'cluster_id'}).set_index('cluster_id')
    print(product_categories_df.info(memory_usage='deep'))

    # get only the cluster ids needed for the train-test categories
    category_cluster_ids = product_categories_df[['category']]# [product_categories_df.category.isin(TRAIN_TEST_CATEGORIES)]
    print(category_cluster_ids.info(memory_usage='deep'))
    category_cluster_ids.to_csv('category_cluster_ids.csv')
    del(product_categories_df)
    gc.collect()
    print(psutil.virtual_memory())

#######################################
#### Clean & Tidy the Offers data ####
start = datetime.now()
try:
    os.chdir('D:/Documents/Large-Scale Product Matching/')
    category_offers_df = reduce_mem_usage(pd.read_csv('category_offers_df.csv', index_col='offer_id'))
    print(category_offers_df.info(memory_usage='deep'))
except Exception as e:
    print(e)
else:

    # get offers data for train-test categories
    os.chdir('D:/Documents/Large-Scale Product Matching/')
    offers_reader = pd.read_json('offers_consWgs_english.json',
                                 orient='records',
                                 chunksize=1e6,
                                 lines=True)

    start = datetime.now()
    offers_df_list = []
    for i, chunk in enumerate(offers_reader):
        print(i)
        offers_df_list.append(reduce_mem_usage(chunk)\
                       .set_index('cluster_id')\
                       .join(category_cluster_ids, how='inner')
                       )
  

    category_offers_df = pd.concat(offers_df_list, axis=0)

    category_offers_df['offer_id'] = category_offers_df.nodeID.str.cat(category_offers_df.url, sep=' ')

    category_offers_df = category_offers_df.reset_index()\
        .set_index('offer_id')\
        .drop(['nodeID', 'url'], axis=1)

    category_offers_df.to_csv('category_offers_df.csv')

print(datetime.now() - start)


category_offers_df.info(memory_usage='deep')
gc.collect()
print(psutil.virtual_memory())

# df = parse_json_column(category_offers_df['parent_schema.org_properties'][:200])

columns_with_json = ['identifiers', 'schema.org_properties', 'parent_schema.org_properties']

parsed_df_list = []
for column in columns_with_json:
    df = parse_json_column(category_offers_df[column], column_prefix=column)
    parsed_df_list.append(df)


parsed_df = reduce_mem_usage(pd.concat(parsed_df_list, sort=False))
parsed_df.info(memory_usage='deep')
parsed_df.to_csv('parsed_df.csv')

print(datetime.now() - start)

# check the dupe gtins
parsed_df_list[0].filter(regex='gtin').notnull().sum(axis=1).value_counts()
parsed_df_list[0][parsed_df_list[0].filter(regex='gtin').notnull().sum(axis=1) > 1].head()


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
# pd.set_option('display.max_columns', 500)
# offers_df.info()
# # offers_df.head()
