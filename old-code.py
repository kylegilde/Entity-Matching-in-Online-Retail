train_test_df_features.index
        .sort_index(axis=0)\
train_test_offers_df

    # save the output
    print('Saving as CSV...')
    train_test_df_features.to_csv('train_test_df_features.csv')
os.getcwd()
print(train_test_df_features.describe(include='all'))
print(train_test_df_features.info(memory_usage='deep'))
calculate_percent_nulls(train_test_df_features)

train_test_df_features.groupby(['filename', 'filename_1', 'filename_2']).size()



if 'train_test_offers_df.csv' in os.listdir():
    train_test_offers_df = reduce_mem_usage(pd.read_csv('train_test_offers_df.csv',
                                                      index_col='offer_id'))


else:
    # From category_offers_df, select the train_test offers and drop completely empty colums
    train_test_offers_df = category_offers_df.join(train_test_offer_ids, how='inner')\
        .dropna(axis=1, how='all')\
        .drop_duplicates()

    train_test_offers_df.loc[train_test_offers_df.index.duplicated(keep=False), :].sort_index(axis=0)

    # save the output
    print('Saving as CSV...')
    train_test_offers_df.to_csv('train_test_offers_df.csv', index_label='offer_id')

print(train_test_offers_df.describe(include='all'))
print(train_test_offers_df.info(memory_usage='deep'))
calculate_percent_nulls(train_test_offers_df)
######################################
Parse the remaining JSON in 3 columns
######################################
Read in Chunked CSV
start = datetime.now()


category_offers_df = pd.read_csv('category_offers_df.csv', chunksize=5e5)
type(category_offers_df)

columns_with_json = ['identifiers', 'schema.org_properties', 'parent_schema.org_properties']
columns_to_drop = ['parent_NodeID', 'relationToParent'] + columns_with_json

new_df_list = []
for i, chunk in enumerate(category_offers_df):
    # Parse each column
    print(i)
    #chunk = next(category_offers_df)
    parsed_df_list = []
    # json_columns_df = clean_json_columns(chunk.loc[:, columns_with_json])
    json_columns_df = clean_json_columns(chunk[columns_with_json])
    for column in columns_with_json:
        print(column)
        df = parse_json_column(json_columns_df[column], column_prefix=column)\
            .reset_index(drop=True)
        # coalesce the gtins
        if column == 'identifiers':
            df = coalesce_gtin_columns(df)
        parsed_df_list.append(df)
        #sys.getsizeof(parsed_df_list)
    # Drop the 3 parsed columns
    chunk.drop(columns_to_drop, axis=1, inplace=True)
    # Concatenate the chunk to the 3 parsed columns & add it to the df list
    parsed_df_list.append(chunk)
    new_df = pd.concat(parsed_df_list, axis=1, sort=False)
    new_df_list.append(new_df)


# my_df = reduce_mem_usage(pd.concat(new_df_list, axis=0))
print('Saving as CSV...')
reduce_mem_usage(pd.concat(new_df_list, axis=0)).to_csv('category_offers_df_v2.csv')
print(datetime.now() - start)


my_df[['schema.org_properties_price', 'parent_schema.org_properties_price']]
new_price = my_df['parent_schema.org_properties_price'].str.replace('(usd|,|gbp)', '')\
    .str.strip()\
    .str.replace(' ', '.')\
    .astype('float')

new_price.value_counts()


gc.collect()
print(psutil.virtual_memory())
parsed_df_list = []


gc.collect()

print(datetime.now() - start)


parsed_df.info(memory_usage='deep')
print('Saving as CSV...')
parsed_df.to_csv('parsed_df.csv')







Load the Offers
os.chdir('D:/Documents/Large-Scale Product Matching/')
offers_df = pd.read_json('offers_consWgs_english.json',
                         orient='records',
                         chunksize=1e4,
                         lines=True)

one_chunk = next(offers_df)



one_chunk.columns
one_chunk.info()
json_normalize(list(one_chunk['identifiers'][:1000]))

json_normalize(one_chunk['identifiers'][:10])

a = one_chunk['identifiers'][:10].apply(lambda x: json_normalize(x))


json_normalize(one_chunk['identifiers'][:10], meta=['/mpn', '/sku', '/productID', '/gtin8', '/gtin13'])
one_chunk.head()

# read json using chunks

df2 = pd.concat([pd.DataFrame(json_normalize(x)) for x in one_chunk['schema.org_properties']], ignore_index=True)
df2.columns = df2.columns.str.strip('/')

new_df = pd.concat([pd.DataFrame(flatten(x)) for x in one_chunk['identifiers']], ignore_index=True)

flatten(dict(one_chunk['identifiers']))

new_df.columns = new_df.columns.str.strip('/')

type(new_df['mpn'])



offers_file_path = 'D:/Documents/Large-Scale Product Matching/samples/sample_offersenglish.json'
with open('offers_consWgs_english.json', encoding="utf8") as offers_file:
    offers_json = [flatten(json.loads(record)) for record in offers_file]
offers_file.close()
gc.collect()
print('file closed')

with open('flattened_offers.json', 'w', encoding="utf8") as flattened_offers:
    json.dump(offers_json, flattened_offers)
https://stackoverflow.com/a/46482086/4463701
flattened_offers.close()
# del(offers_json)
os.chdir('D:/Documents/Large-Scale Product Matching/')
with open('flattened_offers.json', encoding="utf8") as flattened_offers:
    offers_json = [dict(json.loads(record)) for record in flattened_offers]
    pd.DataFrame(offers_json).to_csv('offers_df.csv', index=False)


offers_df.to_csv('D:/Documents/Large-Scale Product Matching/offers_df.csv', index=False)
offers_df = pd.concat(offers_json, ignore_index=True)
offers_df = pd.DataFrame([flatten(record) for record in offers_json])

offers_df.info()
# offers_df.head()
def clean_json_columns(df, removal_class=CHARS_TO_REMOVE_CLASS):
    """

    :param df:
    :param removal_class:
    :return:
    """
    for col in df.columns:
        try:
            df.loc[:, col] = df.loc[:, col].str.replace(removal_class, '')\
                .str.replace("'", '"')\
                .apply(load_and_normalize_json)
        except Exception as e:
            print('clean_json_columns:', e)

    return df
def load_and_normalize_json(a_string):
    try:
        return json_normalize(json.loads(a_string))
    except Exception as e:
        print(e)
        print(a_string)
