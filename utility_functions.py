from urllib.parse import urlparse
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype


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


def calculate_percent_nulls(df):
    """

    :param df:
    :return:
    """
    percentages = df.isnull().sum() / len(df) * 100

    print(percentages.sort_values(ascending=False))
