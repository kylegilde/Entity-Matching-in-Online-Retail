# !/usr/bin/env/python365
"""
Created on Feb 10, 2019
@author: Kyle Gilde
"""
import os
import re
import gc
import timeit
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.io.json import json_normalize
from flatten_json import flatten
import json

def read_train_test_files(file_dir, delimitor='_', category_position=0):
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
        df = pd.read_csv(file, names=['ID_1', 'ID_2', 'label'], sep = '#####', engine='python')
        df['category'] = re.split(delimitor, file)[category_position]
        df_list.append(df)

    return pd.concat(df_list, axis = 0, ignore_index = True)


# Load Train Data
train_df = read_train_test_files('D:/Documents/Large-Scale Product Matching/training-data/')

plt.gcf().clear()
train_df.category.value_counts().plot.bar()



# Load Test Data
test_df = read_train_test_files('D:/Documents/Large-Scale Product Matching/test-data/',
                                category_position=1)

train_test_IDs = pd.concat([test_df.ID_1, test_df.ID_2, train_df.ID_1, train_df.ID_2], axis=0,
                           ignore_index=True)

plt.gcf().clear()
test_df.category.value_counts().plot.bar()


gc.collect()

# Load the Offers
# read json using chunks
os.chdir('D:/Documents/Large-Scale Product Matching/')


# offers_file_path = 'D:/Documents/Large-Scale Product Matching/samples/sample_offersenglish.json'
with open('offers_consWgs_english.json', encoding="utf8") as offers_file:
    offers_json = [flatten(json.loads(record)) for record in offers_file]
offers_file.close()
gc.collect()
print('file closed')

with open('flattened_offers.json', 'w', encoding="utf8") as flattened_offers:
    json.dump(offers_json, flattened_offers)
# https://stackoverflow.com/a/46482086/4463701
flattened_offers.close()
del offers_json
gc.collect()
print('deleted offers_json')


start = timeit.default_timer()
os.chdir('D:/Documents/Large-Scale Product Matching/')
with open('flattened_offers.json', encoding="utf8") as flattened_offers:
    offers_json = [dict(json.loads(record)) for record in flattened_offers]
    pd.DataFrame(offers_json).to_csv('offers_df.csv', index=False)

stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)
sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))


start = timeit.default_timer()
offers_df = pd.read_json('flattened_offers.json',
                       lines=True,
                       encoding='utf8',
                       chunksize=100)



for chunk in offers_df:
    print(chunk.shape)
    #small_df = chunk
    break


type(offers_df)
offers_df.info
stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)
sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))


offers_df.columns
offers_df.head
offers_df['parent_schema.org_properties'][:10]
offers_df.head()
offers_df



import sys
sys.getsizeof(offers_df)


# offers_df.to_csv('D:/Documents/Large-Scale Product Matching/offers_df.csv', index=False)
# offers_df = pd.concat(offers_json, ignore_index=True)
#offers_df = pd.DataFrame([flatten(record) for record in offers_json])
# pd.set_option('display.max_columns', 500)
# offers_df.info()
# # offers_df.head()
