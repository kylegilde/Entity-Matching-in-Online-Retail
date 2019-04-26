
####################### create the specs df ########################

# I ulimately decided against using these as feature because they are incredibly sparse

from json_parsing_functions import *

DATA_DIRECTORY = 'D:/Documents/Large-Scale Product Matching/'

if 'specs_df.csv' in os.listdir():
    train_test_offer_specs = reduce_mem_usage(pd.read_csv('specs_df.csv'))
    calculate_percent_nulls(train_test_offer_specs)

else:
    offer_specs_reader = pd.read_json('specTablesConsistent', lines=True, orient='records',
                                      chunksize=1e6)
    # Load Train & Test Data
    train_df, test_df = read_train_test_files(DATA_DIRECTORY + '/training-data/'),\
                        read_train_test_files(DATA_DIRECTORY + '/test-data/')

    train_test_offer_ids = rbind_train_test_offers(train_df, test_df)

    unique_train_test_offer_ids = pd.DataFrame({'offer_id' : train_test_offer_ids.index.unique()}).set_index('offer_id')


    temp_df = unique_train_test_offer_ids\
        .reset_index()

    unique_train_test_offer_urls = temp_df.offer_id.str.split(' ', 1, expand=True)\
        .rename(columns={1:'url'})\
        .drop(0, axis=1)\
        .set_index('url')

    unique_train_test_offer_urls['offer_id'] = temp_df.offer_id.values

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


    specs_df = reduce_mem_usage(pd.concat(new_df_list, sort=True))
    #specs_df.columns = pd.Series(specs_df.columns).apply(clean_text) #.str.replace('[^\x00-\x7F]','')

    specs_df.info(memory_usage='deep')
    # Save df
    print('Saving as CSV...')
    specs_df.to_csv('specs_df.csv', index_label='offer_id')
