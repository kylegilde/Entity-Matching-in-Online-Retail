# !/usr/bin/env/python365
"""
Created on Feb 10, 2019
@author: Kyle Gilde
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt
from json_parsing_functions import *

# Initialize some constants
TRAIN_TEST_CATEGORIES = ['Computers_and_Accessories', 'Camera_and_Photo', 'Shoes', 'Jewelry']
PRICE_COLUMN_NAMES = ['price','parent_price']
DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'
os.chdir(DATA_DIRECTORY)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

####################################################################
################### Load Train and Test Data #######################
####################################################################

# Load Train & Test Data
train_df, test_df = read_train_test_files(DATA_DIRECTORY + '/training-data/'),\
                    read_train_test_files(DATA_DIRECTORY + '/test-data/')

print(train_df.info(memory_usage='deep'))
print(test_df.info(memory_usage='deep'))

plt.gcf().clear()
train_df.filename.value_counts().plot.bar()

train_test_df = pd.concat([train_df, test_df], axis=0)
print(train_test_df.info())
train_test_df = create_file_categories(train_test_df)
train_test_df.to_csv('train_test_df.csv')

train_test_offer_ids = rbind_train_test_offers(train_df, test_df)
print(train_test_offer_ids.info())

unique_train_test_offer_ids = pd.DataFrame({'offer_id' : train_test_offer_ids.index.unique()}).set_index('offer_id')
print(len(unique_train_test_offer_ids))


####################################################################
################# Clean & Tidy the Offers data #####################
####################################################################

# if file exists read it, otherwise create it
if 'train_test_offer_features.csv' in os.listdir():
    train_test_offer_features = reduce_mem_usage(pd.read_csv('train_test_offer_features.csv',
                                                      index_col='offer_id'))
    print(train_test_offer_features.info(memory_usage='deep'))
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
                    df = parse_price_columns(df, PRICE_COLUMN_NAMES)

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
    train_test_offer_features = reduce_mem_usage(pd.concat(offers_df_list, axis=0)\
                                          .drop(more_cols_to_drop, axis=1)\
                                          .dropna(axis=1, how='all'))

    train_test_offer_features.to_csv('train_test_offer_features.csv', index_label='offer_id')

    print(train_test_offer_features.describe(include='all'))
    print(train_test_offer_features.info(memory_usage='deep'))
    calculate_percent_nulls(train_test_offer_features)

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

    train_test_offer_features = train_test_offer_features\
        .reset_index()\
        .set_index('cluster_id', drop=False)\
        .join(pd.concat(english_cluster_list, axis=0))\
        .set_index('offer_id')

    #
    train_test_offer_features = create_file_categories[train_test_offer_features]

    # Save df
    print('Saving as CSV...')
    train_test_offer_features.to_csv('train_test_offer_features.csv', index_label='offer_id')
    # missingness plot
    # sns.heatmap(train_test_offer_features[['brand', 'manufacturer']].isnull(), cbar=False)

    calculate_percent_nulls(train_test_offer_features)
    train_test_offer_features.domain.value_counts()
    train_test_offer_features.brand.value_counts()



###################################################################
#### Create the df for only the offers in train/test set ##########
###################################################################

# if file exists read it, otherwise create it
if 'train_test_feature_pairs.csv' in os.listdir():
    train_test_feature_pairs = reduce_mem_usage(pd.read_csv('train_test_feature_pairs.csv'))

else:
    # join the offer details to the pair of offer ids
    # and add suffixes to them
    # move label column to 1st position
    # drop completely null columns
    train_test_feature_pairs = train_test_df.set_index('offer_id_1', drop=False)\
        .join(train_test_offer_features.add_suffix('_1'), how='left')\
        .set_index('offer_id_2', drop=False)\
        .join(train_test_offer_features.add_suffix('_2'), how='left')\
        .sort_index(axis=1)\
        .set_index('label')\
        .reset_index()

    train_test_feature_pairs.to_csv('train_test_feature_pairs.csv', index=False)

    calculate_percent_nulls(train_test_feature_pairs)






