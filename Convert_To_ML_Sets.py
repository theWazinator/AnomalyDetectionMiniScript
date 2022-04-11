"""
 Divide the aggregate dataframes into train, validation, and test sets.
 All invalid records are dropped and two copies of the validation and test sets are made
 One copy has both the unclean and clean (presumed uncensored) data, and the other only the clean data
 """

import os
from Convert_To_ML_Helper_Methods import *
from add_GFWatch_column import run_add_GFWatch_column


# remove invalid records
def remove_invalid_records(df):

    valid_df = df[(df.in_control_group == True) & (df.control_response_start_success == True) & (df.excluded_below_threshold == False)]

    return valid_df.copy()

def get_ASN_list(input_df, row):

    IP_count = input_df['test_response_IP_count'].iloc[row]

    asn_list = []

    for index in range(0, IP_count):

        asn_list.append(input_df['test_response_' +str(index)+ '_asnum'].iloc[row])

    return asn_list

def remove_unclean_records (input_df, AS_count_df):

    # Convert AS_count_df into dictionary

    domain_datetime_clean_dict = {}
    domain_datetime_asn_dict = {}

    for row in range(0, AS_count_df.shape[0]):

        datetime = AS_count_df['datetime'].iloc[row].strftime('%Y-%m-%d-%H-%M-%S')
        domain = AS_count_df['domain_name'].iloc[row]
        clean_records = AS_count_df['clean_records'].iloc[row]
        asn = AS_count_df['most_common_asn_num'].iloc[row]

        if domain not in domain_datetime_clean_dict.keys():

            domain_datetime_clean_dict[domain] = {datetime: clean_records}
            domain_datetime_asn_dict[domain] = {datetime: asn}

        else:

            domain_datetime_clean_dict[domain][datetime] = clean_records
            domain_datetime_asn_dict[domain][datetime] = asn

    indices_to_remove = []

    for row in range(0, input_df.shape[0]):

        datetime = input_df['batch_datetime'].iloc[row].strftime('%Y-%m-%d-%H-%M-%S')
        domain = input_df['test_url'].iloc[row]
        control_response_end_success = input_df['control_response_end_success'].iloc[row]
        anomaly = input_df['anomaly'].iloc[row]
        connect_error = input_df['connect_error'].iloc[row]
        test_query_successful = input_df['test_query_successful'].iloc[row]

        remove_index = True

        asn_list = get_ASN_list(input_df, row)

        # Check if domain and datetime combination is in the ASN list
        if domain in domain_datetime_clean_dict.keys() and datetime in domain_datetime_clean_dict[domain].keys():

            # Check ASN list quality control
            if domain_datetime_clean_dict[domain][datetime] == "CLEAN":

                if control_response_end_success == True and anomaly == False and connect_error == False and test_query_successful == True:

                    if domain_datetime_asn_dict[domain][datetime] in asn_list:

                        remove_index = False # If there is a problem here, it is probably because the asn column is not recognized as numerical by Pandas

        if remove_index == True:

            indices_to_remove.append(row)

    clean_df = input_df.drop(index=indices_to_remove)

    return clean_df.copy()

# Create the training data from the original dataframe
def create_ML_ready_data(df, AS_count_df, clean_only, index_list, save_filename, data_type):

    df = df.iloc[index_list[0]: index_list[1]].copy() # Select only the rows we want

    total_records_count = df.shape[0]

    if clean_only == False:

        clean_moniker = "Mixed"

    if clean_only == True:

        df = remove_unclean_records(df, AS_count_df)

        clean_moniker = "Clean"

    records_removed_count = total_records_count - df.shape[0]

    df['anomaly'].astype(int).to_csv(path_or_buf=save_filename+data_type+ "_" +clean_moniker+ "_targetFeature_anomaly.csv", \
             index=False) # Converts True and False to 1 and 0

    df.drop(['anomaly'], axis=1)

    if country_code == 'CN':

        df['GFWatch_Censored'].to_csv(path_or_buf=save_filename+data_type+ "_" +clean_moniker+ "_targetFeature_GFWatch_Censored.csv", \
                index=False)

        df.drop(['GFWatch_Censored'], axis=1)

    elif country_code == "US":

        df['Presumed_Censored'].to_csv(path_or_buf=save_filename+data_type+ "_" +clean_moniker+ "_targetFeature_Presumed_Censored.csv", \
                index=False)

        df.drop(['Presumed_Censored'], axis=1)

    else:

        pass # No presumption of censorship for other countries

    df = create_ML_features(df)

    df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=save_filename+data_type+ "_" +clean_moniker+ "_descriptiveFeatures_fullDataset.gzip")

    return total_records_count, records_removed_count

country_code = "CN"
country_name = "China"

training_split_fraction = 0.8
validation_split_fraction = 0.1
testing_split_fraction = 1 - training_split_fraction - validation_split_fraction

home_file_name = r"/home/jambrown/CP_Analysis/"
ml_ready_data_file_name = home_file_name +country_code+ "/ML_ready_dataframes/"
os.mkdir(ml_ready_data_file_name)
aggregate_file_name = home_file_name +country_code+ "/raw_dataframe.gzip"

GFWatch_table_filename = r'/home/jambrown/CP_Analysis/gfwatch_censored_domains.csv'
AS_count_table_filename = home_file_name + "max_asn_aggregate.gzip"

AS_count_df = pd.read_parquet(path=AS_count_table_filename, engine='pyarrow')

original_df = pd.read_parquet(path=aggregate_file_name, engine='pyarrow')

original_df_length = original_df.shape[0]

valid_df = remove_invalid_records(original_df)

valid_df_length = valid_df.shape[0]

print("Total number of probes: " +str(original_df_length))
print("Valid number of probes: " +str(valid_df_length))

# Calculate indices for dividing data into unique training, validation, and testing components
training_index_list = (0, int(valid_df_length*training_split_fraction))
validation_index_list = (int(valid_df_length*training_split_fraction)+1, int(valid_df_length*(training_split_fraction+validation_split_fraction)))
testing_index_list = (int(valid_df_length*(training_split_fraction+validation_split_fraction))+1, valid_df_length)

if country_code == "CN":

    original_with_newColumn_df = run_add_GFWatch_column(valid_df, GFWatch_table_filename)

elif country_code == "US":

    original_with_newColumn_df = valid_df # change the reference
    original_with_newColumn_df['Presumed_Censored'] = 0  # All US records are presumed uncensored

else:

    original_with_newColumn_df = valid_df  # change the reference and don't add the new column

original_with_newColumn_df = original_with_newColumn_df.sample(frac=1) # Shuffle the dataframe - this is the last time this is done in the pipeline

# TODO uncomment this section
# # Create clean training dataset
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=True,
#                            index_list=training_index_list, save_filename=ml_ready_data_file_name, data_type="TRAINING")
#
# print("Training data (clean) total probes: " +str(total_records_count))
# print("Training data (clean) probes removed: " +str(records_removed_count))
#
# # Create mixed training set
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=False,
#                            index_list=training_index_list, save_filename=ml_ready_data_file_name, data_type="TRAINING")
#
# print("Training data (mixed) total probes: " +str(total_records_count))
# print("Training data (mixed) probes removed: " +str(records_removed_count))
#
# # Create clean validation dataset
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=True,
#                            index_list=validation_index_list, save_filename=ml_ready_data_file_name, data_type="VALIDATION")
#
# print("Validation data (clean) total probes: " +str(total_records_count))
# print("Validation data (clean) probes removed: " +str(records_removed_count))
#
# # Create mixed validation set
# total_records_count, records_removed_count \
#     = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=False,
#                            index_list=validation_index_list, save_filename=ml_ready_data_file_name, data_type="VALIDATION")
#
# print("Validation data (mixed) total probes: " +str(total_records_count))
# print("Validation data (mixed) probes removed: " +str(records_removed_count))

# Create clean testing dataset
total_records_count, records_removed_count \
    = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=True,
                           index_list=testing_index_list, save_filename=ml_ready_data_file_name, data_type="TESTING")

print("Testing data (clean) total probes: " +str(total_records_count))
print("Testing data (clean) probes removed: " +str(records_removed_count))

# Create mixed testing set
total_records_count, records_removed_count \
    = create_ML_ready_data(original_with_newColumn_df, AS_count_df, clean_only=False,
                           index_list=testing_index_list, save_filename=ml_ready_data_file_name, data_type="TESTING")

print("Testing data (mixed) total probes: " +str(total_records_count))
print("Testing data (mixed) probes removed: " +str(records_removed_count))



