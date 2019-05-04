from datetime import datetime
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


def calculate_percent_nulls(df, print_series=True, return_series=False):
    """

    :param df:
    :return:
    """
    percentages = df.isnull().sum() / len(df) * 100
    percentages_sorted = percentages.sort_values(ascending=False)

    if print_series:
        print(percentages_sorted)

    if return_series:
        return(percentages_sorted)


def get_duration_hours(start_time):
    """
    Prints and returns the time difference in hours

    :param start_time: datetime object
    :return: time difference in hours
    """
    time_diff = datetime.now() - start_time
    time_diff_hours = time_diff.seconds / 3600
    print('hours:', round(time_diff_hours, 2))
    return time_diff_hours