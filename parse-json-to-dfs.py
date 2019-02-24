# !/usr/bin/env/python365
"""
Created on Feb 10, 2019
@author: Kyle Gilde
"""
import os
from datetime import datetime

from urllib.parse import urlparse

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
DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
os.chdir(DATA_DIRECTORY)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)


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


def parse_price_columns(df, price_column_names=PRICE_COLUMN_NAMES):
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


def calculate_percent_nulls(df):
    """

    :param df:
    :return:
    """
    print(df.isnull().sum() / len(df) * 100)

##################################
#### Load Train and Test Data ####
##################################
# Load Train & Test Data
train_df, test_df = read_train_test_files(DATA_DIRECTORY + '/training-data/'),\
                    read_train_test_files(DATA_DIRECTORY + '/test-data/')


print(train_df.info(memory_usage='deep'))
print(test_df.info(memory_usage='deep'))

plt.gcf().clear()
train_df.filename.value_counts().plot.bar()

train_test_df = pd.concat([train_df, test_df], axis=0)
print(train_test_df.info())

train_test_offer_ids = rbind_train_test_offers(train_df, test_df)
print(train_test_offer_ids.info())

unique_train_test_offer_ids = pd.DataFrame({'offer_id' : train_test_offer_ids.index.unique()}).set_index('offer_id')
print(len(unique_train_test_offer_ids))


#######################################
#### Clean & Tidy the Offers data ####
#######################################

# if file exists read it, otherwise create it
if 'train_test_offers_df.csv' in os.listdir():
    category_offers_df = reduce_mem_usage(pd.read_csv('train_test_offers_df.csv',
                                                      index_col='offer_id'))
    print(category_offers_df.info(memory_usage='deep'))
else:
    start = datetime.now()
    # Read or Create the Cluster ID-Category Mappings
    # category_cluster_ids = get_cluster_ids()

    # get offers data for train-test categories
    offers_reader = pd.read_json('offers_consWgs_english.json',
                                 orient='records',
                                 chunksize=1e6,
                                 lines=True)

    columns_with_json = ['identifiers', 'schema_org_properties', 'parent_schema_org_properties']
    columns_to_drop = ['parent_NodeID'] + columns_with_json # , 'relationToParent'
    more_cols_to_drop = ['availability']


    offers_df_list = []
    for i, chunk in enumerate(offers_reader):
        print(i)
        # chunk = next(offers_reader)


        # create the unique offer_id
        chunk['offer_id'] = chunk.nodeID.str.cat(chunk.url, sep=' ')
        # inner join on the train-test offer ids
        new_chunk = reduce_mem_usage(chunk)\
                       .set_index('offer_id')\
                       .join(unique_train_test_offer_ids, how='inner')

        # if any rows remain after inner join, parse the data
        if len(new_chunk):
            # clean column names
            new_chunk.columns = new_chunk.columns.str.replace('.', '_')
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
                if np.sum(df.columns.isin(PRICE_COLUMN_NAMES)):
                    df = parse_price_columns(df)

                parsed_df_list.append(df)

            print(datetime.now() - start)
            # Drop the 3 parsed columns
            new_chunk.drop(columns_to_drop, axis=1, inplace=True)
            # Concatenate the chunk to the 3 parsed columns & add it to the df list
            parsed_df_list.append(new_chunk)
            # combine the parent child columns
            new_chunk = coalesce_parent_child_columns(pd.concat(parsed_df_list, axis=1, sort=False))

            # Remove the terms null and description
            new_chunk['name'] = new_chunk.name.combine_first(df.parent_title)\
                .str.replace('^null\s*?,|,\s*?null$', '')
            new_chunk['description'] = new_chunk.description.str.replace('^description', '')
            offers_df_list.append(new_chunk)


    # Combine all the offers & save the output
    print('Saving as CSV...')
    train_test_offers_df = reduce_mem_usage(pd.concat(offers_df_list, axis=0)\
                                          .drop(more_cols_to_drop, axis=1)\
                                          .dropna(axis=1, how='all'))
    train_test_offers_df.to_csv('train_test_offers_df.csv', index_label='offer_id')

    print(train_test_offers_df.describe(include='all'))
    print(train_test_offers_df.info(memory_usage='deep'))
    calculate_percent_nulls(train_test_offers_df)

    ################################
    # Add the Category Attribute
    ################################
    print('Adding the category from clusters file...')
    english_clusters_reader = pd.read_json('clusters_english.json',
                                                    lines=True,
                                                    orient='records',
                                                    chunksize=1e6)
    english_cluster_list = []
    for i, chunk in enumerate(english_clusters_reader):
        print(i)

        another_chunk = chunk[['id', 'category', 'id_values']]\
            .rename(columns={'id':'cluster_id'})\
            .set_index('cluster_id')

        english_cluster_list.append(another_chunk)

    train_test_offers_df = train_test_offers_df\
        .reset_index()\
        .set_index('cluster_id', drop=False)\
        .join(pd.concat(english_cluster_list, axis=0))\
        .set_index('offer_id')

    # Save df
    print('Saving as CSV...')
    train_test_offers_df.to_csv('train_test_offers_df.csv', index_label='offer_id')
    # missingness plot
    # sns.heatmap(train_test_offers_df[['brand', 'manufacturer']].isnull(), cbar=False)




###########################################################
if 'train_test_offer_specs.csv' in os.listdir():
    train_test_offer_specs = reduce_mem_usage(pd.read_csv('train_test_offer_specs.csv'))

else:
    offer_specs_reader = pd.read_json('specTablesConsistent', lines=True, orient='records',
                                      chunksize=1e6)

    temp_df = unique_train_test_offer_ids\
        .reset_index()

    #unique_train_test_offer_urls = pd.DataFrame()
    unique_train_test_offer_urls = temp_df.offer_id.str.split(' ', 1, expand=True)\
        .rename(columns={1:'url'})\
        .drop(0, axis=1)\
        .set_index('url')

    new_df_list = []
    for i, chunk in enumerate(offer_specs_reader):
        print(i)
        # chunk = next(offer_specs_reader)

        new_chunk = chunk.set_index('url')[['keyValuePairs']]\
            .loc[chunk.keyValuePairs.str.len().values > 0, :]\
            .join(unique_train_test_offer_urls, how='inner')

        if len(new_chunk):
            new_df = pd.concat([json_normalize(line) for line in new_chunk.keyValuePairs], sort=True)
            new_df.index = new_chunk.index
            new_df_list.append(new_df)

    specs_df = pd.concat(new_df_list, sort=True)
###################################################################
#### Create the df for only the offers in train/test set ##########
###################################################################
# if file exists read it, otherwise create it

if 'train_test_df_features.csv' in os.listdir():
    train_test_df_features = reduce_mem_usage(pd.read_csv('train_test_df_features.csv'))

else:
    # join the offer details to the pair of offer ids
    # and add suffixes to them
    # move label column to 1st position
    # drop completely null columns
    train_test_df_features = train_test_df.set_index('offer_id_1', drop=False)\
        .join(train_test_offers_df.add_suffix('_1'), how='left')\
        .set_index('offer_id_2', drop=False)\
        .join(train_test_offers_df.add_suffix('_2'), how='left')\
        .sort_index(axis=1)\
        .set_index('label')\
        .reset_index()

    train_test_df_features.to_csv('train_test_df_features.csv', index=False)

    calculate_percent_nulls(train_test_df_features)






#train_test_df_features.index
        #.sort_index(axis=0)\
# train_test_offers_df
#
#     # save the output
#     print('Saving as CSV...')
#     train_test_df_features.to_csv('train_test_df_features.csv')
# os.getcwd()
# print(train_test_df_features.describe(include='all'))
# print(train_test_df_features.info(memory_usage='deep'))
# calculate_percent_nulls(train_test_df_features)
#
# train_test_df_features.groupby(['filename', 'filename_1', 'filename_2']).size()
#
#
#
# if 'train_test_offers_df.csv' in os.listdir():
#     train_test_offers_df = reduce_mem_usage(pd.read_csv('train_test_offers_df.csv',
#                                                       index_col='offer_id'))
#
#
# else:
#     # From category_offers_df, select the train_test offers and drop completely empty colums
#     train_test_offers_df = category_offers_df.join(train_test_offer_ids, how='inner')\
#         .dropna(axis=1, how='all')\
#         .drop_duplicates()
#
#     train_test_offers_df.loc[train_test_offers_df.index.duplicated(keep=False), :].sort_index(axis=0)
#
#     # save the output
#     print('Saving as CSV...')
#     train_test_offers_df.to_csv('train_test_offers_df.csv', index_label='offer_id')
#
# print(train_test_offers_df.describe(include='all'))
# print(train_test_offers_df.info(memory_usage='deep'))
# calculate_percent_nulls(train_test_offers_df)
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
