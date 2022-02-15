#
# Transformed compressed Satellite data files into a machine-learning usable format
# The program assumes that all the tar files have already been downloaded using CP_crawler.py
# Before starting, the following folders need to be created:
# 1. Folder for temporarily holding unzipped tar files and segmented JSON folders (the contents of this folder are deleted after they have been processed)
# 2. Folder containing subfolders coresponding to each scan, with each containing
#    a. a folder holding blockpages.json, dns.pkt, resolvers.json
#    b. a folder holding unmodified dataframes (compressed) extracted from results.json
#    c. a folder holding unmodified dataframes (compressed) derived from the unmodified dataframes, one for each country being examined
#    d. a folder holding ML-ready dataframes (compressed) derived from the unmodified dataframes, one for each country being examined
#

import os.path
import pandas as pd
import datetime as dt
from methods import *
import random
import copy
import pickle
from sklearn.preprocessing import StandardScaler
import shutil
import tarfile

# Get list of files in folder - TODO Change these names for your file structure
cp_downloads_zipped_file_name = r"/home/jambrown/CP_Downloads/"
temp_file_name = r"/home/jambrown/temp_working_folder_unzipped_files/"
extractions_file_name = r"/home/jambrown/CP_Extractions/"
countries_to_select = ["United States", "China", "India", "Russia", "Turkey", "Iran"]

list_of_zipped_cp_files = os.listdir(cp_downloads_zipped_file_name)
list_of_zipped_cp_files.sort(reverse=True) # Ensures most recent scans are processed first

list_of_zipped_cp_files = list_of_zipped_cp_files[:1] # TODO remove this line

for zipped_cp_file in list_of_zipped_cp_files:

    cp_scan_only_name = zipped_cp_file.split('.')[0]

    print("Beginning to process file: " +str(cp_scan_only_name))

    # Create (temporary) subdirectory for split JSON files
    json_divide_file_name = temp_file_name + r'json_split_files/'
    os.mkdir(json_divide_file_name)

    # Create new scan-specific folder and sub-directory
    cp_scan_file_name = extractions_file_name+cp_scan_only_name+r"/"
    os.mkdir(cp_scan_file_name)

    # Create subfolder containing the other docs extracted (blockpages.json, dns.pkt, resolvers.json, resolvers_raw.json), as well as a dictionary of vantage points
    other_docs_file_name = cp_scan_file_name + r"other_docs/"
    os.mkdir(other_docs_file_name)

    # Create subfolder containing the raw dataframes extracted from results.json
    dataframe_raw_aggregate_file_name = cp_scan_file_name + r"dataframes_raw_aggregate/"
    os.mkdir(dataframe_raw_aggregate_file_name)

    # Create subfolder containing the raw country-specific dataframes extracted from results.json
    dataframe_raw_countrylevel_file_name = cp_scan_file_name + r"dataframes_raw_country-level/"
    os.mkdir(dataframe_raw_countrylevel_file_name)

    # Create subfolder containing V1 of the ML-ready dataframes derived from the raw dataframes
    dataframe_ML_file_name = cp_scan_file_name + r"dataframes_ML_V1/"
    os.mkdir(dataframe_ML_file_name)

    # Extract files to a temporary folder
    print("Beginning extraction of file " +str(cp_scan_only_name))
    tar = tarfile.open(zipped_cp_file, "r:gz")
    tar.extractall(path=temp_file_name)
    tar.close()
    print("Finished extraction of file " + str(cp_scan_only_name))

    # Move other items (blockpages.json, dns.pkt, resolvers.json, resolvers_raw.json) to new location
    shutil.move(temp_file_name + r"blockpages.json", other_docs_file_name)
    shutil.move(temp_file_name + r"dns.pkt", other_docs_file_name)
    shutil.move(temp_file_name + r"resolvers.json", other_docs_file_name)
    shutil.move(temp_file_name + r"resolvers_raw.json", other_docs_file_name)

    # Begin splitting the results.json file into manageable parts

    og_JSON_download_filename = temp_file_name + r"results.json"
    json_splitfile_output_filename_base = r'_JSON_Part.txt'
    records_per_file = 250000

    # Divide results.json into manageable parts
    DivideJSON(og_JSON_download_filename, \
               json_divide_file_name, \
               json_splitfile_output_filename_base, \
               records_per_file)

    # Create dataframe with all features
    print("Create raw dataframe")
    batch_dt_input = format_datetime_from_file_name(cp_scan_only_name) # the starting time for this batch of probes as a string
    splitfile_count = len(os.listdir(json_divide_file_name))

    for fileNumber in range(0, splitfile_count):

        json_split_filename = json_divide_file_name + str(fileNumber) + json_splitfile_output_filename_base

        df = json_to_df(json_split_filename, batch_dt_input)

        # fastparquet must be used so that the categorical variables associated with integers (ex rcode) will also deserialize into categorical variables
        df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=(dataframe_raw_aggregate_file_name + str(splitfile_count) + "_raw_dataframe.gzip"))
        print("Parquet file size in bytes with gzip compression: " + str(os.path.getsize(dataframe_raw_aggregate_file_name + str(splitfile_count) + "_raw_dataframe.gzip")))
        print("Finished file " +str(fileNumber), flush=True)

    # Create complete list of vantage points used for each location
    print("Create list of vantage points")
    max_vantagepoint_count_per_country = 100
    vantagepoint_dictionary_filename = other_docs_file_name + "vantagepoint_dictionary.txt" # Save dictionary in other docs file for later use

    country_list = {}

    for fileNumber in range(0, splitfile_count):

        df = pd.read_parquet(path=dataframe_raw_aggregate_file_name + str(splitfile_count) + "_raw_dataframe.gzip", engine='pyarrow')

        countries = df["country_name"]
        vps = df["vantage point"]

        index = 0
        for country in countries:

            if country not in country_list.keys():
                country_list[country] = {}

            country_list[country][vps[index]] = None

            index += 1

        print("Finished file " + str(fileNumber), flush=True)

    # Shuffle list order
    final_country_list = {}

    for country in country_list.keys():
        temp_list = list(country_list[country].keys())
        random.shuffle(temp_list)
        final_country_list[country] = temp_list

    # Save dictionary object
    with open(vantagepoint_dictionary_filename, "wb") as file:
        pickle.dump(final_country_list, file)

    country_vp_list_dict = {}

    # Get the list of VPs for each country
    with open(vantagepoint_dictionary_filename, "rb") as file:
        long_country_vp_list_dict = pickle.load(file)

    # Get a list of vantage points used in each country (list was shuffled in previous function)
    for country in long_country_vp_list_dict.keys():
        vp_list = []

        for i in range(0, min(max_vantagepoint_count_per_country, len(long_country_vp_list_dict[country]))):
            vp_list.append(long_country_vp_list_dict[country][i])

        country_vp_list_dict[country] = vp_list

    # Extract dataframe column information from first dataframe
    df_0 = pd.read_parquet(path=dataframe_raw_aggregate_file_name + str(0) + "_raw_dataframe.gzip", engine='pyarrow')
    master_df = pd.DataFrame.from_records(data=df_0, nrows=1)
    master_df.drop(labels=master_df.index[0:], inplace=True)

    for country in countries_to_select:

        for fileNumber in range(0, splitfile_count):

            partial_df = pd.read_parquet(path=dataframe_raw_aggregate_file_name + str(splitfile_count) + "_raw_dataframe.gzip", engine='pyarrow')

            partial_df = partial_df[partial_df.country_name.isin([country]) == True]

            # Select only the vantage points we want
            partial_df = partial_df[((partial_df.country_name.isin([country]) == True) & \
                                     (partial_df.vantage_point.isin(country_vp_list_dict[country]) == True)) ]

            # TODO Note: no longer removing false anomalies from US control data set
            # if country == "United States":
            #     partial_df = partial_df[~((partial_df.country_name.isin(["United States"]) == True) & partial_df.anomaly)]

            # Merge with master dataframe
            master_df = pd.concat([master_df, partial_df], ignore_index=True)

            print("Finished file " + str(fileNumber), flush=True)

        # Shuffle records
        master_df = master_df.sample(frac=1)

        # Relabel Records
        master_df.reset_index(drop=True, inplace=True)

        # Save file
        # fastparquet must be used so that the categorical variables associated with integers (ex rcode) will also deserialize into categorical variables
        master_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=dataframe_raw_countrylevel_file_name + country + ".gzip")

        # # print to CSV for viewing
        # master_df.to_csv(path_or_buf=r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/SmallSample.csv", \
        #              index=True)

        # Create feature table
        # TODO remove these file names
        anomaly_vector_file_name = other_docs_file_name + "anomaly_vector.txt" # Save anomaly records in other docs file for later use
        standard_scaler_file_name = other_docs_file_name + "anomaly_vector.txt" # Save standard scaler in other docs file for later use.
        # records_file_complete = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/RecordsSelected.gzip"
        # output_path_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ReadyToGoDataset"


        columns_to_keep = [
            'average_matchrate',
            'untagged_controls',
            'untagged_response',
            'passed_liveness',
            'connect_error',
            'in_control_group',
            'anomaly',
            'excluded',
            'excluded_is_CDN',
            'excluded_below_threshold',
            'delta_time',
            'control_response_start_success',
            'control_response_end_success',
            'control_response_start_has_type_a',
            'control_response_start_rcode',
            'control_response_end_has_type_a',
            'control_response_end_rcode',
            'test_query_successful',
            'test_query_unsuccessful_attempts',
            'test_noresponse_1_has_type_a',
            'test_noresponse_1_rcode',
            'test_noresponse_2_has_type_a',
            'test_noresponse_2_rcode',
            'test_noresponse_3_has_type_a',
            'test_noresponse_3_rcode',
            'test_noresponse_4_has_type_a',
            'test_noresponse_4_rcode',
            'test_response_has_type_a',
            'test_response_rcode',
            'test_response_IP_count',
            'test_response_0_IP_match',
            'test_response_0_http_match',
            'test_response_0_cert_match',
            'test_response_0_asnum_match',
            'test_response_0_asname_match',
            'test_response_0_match_percentage',
            'test_response_1_IP_match',
            'test_response_1_http_match',
            'test_response_1_cert_match',
            'test_response_1_asnum_match',
            'test_response_1_asname_match',
            'test_response_1_match_percentage',
            'test_response_2_IP_match',
            'test_response_2_http_match',
            'test_response_2_cert_match',
            'test_response_2_asnum_match',
            'test_response_2_asname_match',
            'test_response_2_match_percentage',
            'test_response_3_IP_match',
            'test_response_3_http_match',
            'test_response_3_cert_match',
            'test_response_3_asnum_match',
            'test_response_3_asname_match',
            'test_response_3_match_percentage',
            'test_response_4_IP_match',
            'test_response_4_http_match',
            'test_response_4_cert_match',
            'test_response_4_asnum_match',
            'test_response_4_asname_match',
            'test_response_4_match_percentage',
        ]

        # Open records file, keeping only columns we need
        df = pd.read_parquet(path=records_file_complete, engine='pyarrow', columns=columns_to_keep)

        # Save anomaly records seprately (note: test records should not be shuffled from this point)
        anomaly_vector = df['anomaly'].to_numpy().copy()
        with open(anomaly_vector_file_name, "wb") as file:
            pickle.dump(anomaly_vector, file)

        # Drop anomaly column
        df = df.drop("anomaly", axis="columns")

        # Create new boolean columns to indicate more IPs
        row_count = df.shape[0]

        ip_count = df['test_response_IP_count']

        more_ips = np.full(shape=row_count, dtype=bool, fill_value=False)

        for index in range(0, row_count):
            if ip_count.iloc[index] > 5:
                more_ips[index] = True

        df['more_IPs'] = pd.Series(more_ips)

        #  Create new boolean columns to indicate if IP is being used

        include_IP_0 = np.full(shape=row_count, dtype=bool, fill_value=False)
        include_IP_1 = np.full(shape=row_count, dtype=bool, fill_value=False)
        include_IP_2 = np.full(shape=row_count, dtype=bool, fill_value=False)
        include_IP_3 = np.full(shape=row_count, dtype=bool, fill_value=False)
        include_IP_4 = np.full(shape=row_count, dtype=bool, fill_value=False)

        for index in range(0, row_count):
            if ip_count.iloc[index] == 1:
                include_IP_0[index] = True

            if ip_count.iloc[index] == 2:
                include_IP_1[index] = True

            if ip_count.iloc[index] == 3:
                include_IP_2[index] = True

            if ip_count.iloc[index] == 4:
                include_IP_3[index] = True

            if ip_count.iloc[index] == 5:
                include_IP_4[index] = True

        df['include_IP_0'] = pd.Series(include_IP_0)
        df['include_IP_1'] = pd.Series(include_IP_1)
        df['include_IP_2'] = pd.Series(include_IP_2)
        df['include_IP_3'] = pd.Series(include_IP_3)
        df['include_IP_4'] = pd.Series(include_IP_4)

        # Change IP match percentage rate to same as average if IP is not being used

        new_test_response_0_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)
        new_test_response_1_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)
        new_test_response_2_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)
        new_test_response_3_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)
        new_test_response_4_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)

        old_test_response_0_match_percentage = df['test_response_0_match_percentage']
        old_test_response_1_match_percentage = df['test_response_1_match_percentage']
        old_test_response_2_match_percentage = df['test_response_2_match_percentage']
        old_test_response_3_match_percentage = df['test_response_3_match_percentage']
        old_test_response_4_match_percentage = df['test_response_4_match_percentage']

        average_matchrate = df['average_matchrate']

        for index in range(0, row_count):
            if old_test_response_0_match_percentage.iloc[index] < 0: # if empty IP, should be -2 in float notation if
                new_test_response_0_match_percentage[index] = average_matchrate[index]

            if old_test_response_1_match_percentage.iloc[index] < 0: # if empty IP, should be -2 in float notation if
                new_test_response_1_match_percentage[index] = average_matchrate[index]

            if old_test_response_2_match_percentage.iloc[index] < 0: # if empty IP, should be -2 in float notation if
                new_test_response_2_match_percentage[index] = average_matchrate[index]

            if old_test_response_3_match_percentage.iloc[index] < 0: # if empty IP, should be -2 in float notation if
                new_test_response_3_match_percentage[index] = average_matchrate[index]

            if old_test_response_4_match_percentage.iloc[index] < 0: # if empty IP, should be -2 in float notation if
                new_test_response_4_match_percentage[index] = average_matchrate[index]

        df['test_response_0_match_percentage'] = pd.Series(new_test_response_0_match_percentage)
        df['test_response_1_match_percentage'] = pd.Series(new_test_response_1_match_percentage)
        df['test_response_2_match_percentage'] = pd.Series(new_test_response_2_match_percentage)
        df['test_response_3_match_percentage'] = pd.Series(new_test_response_3_match_percentage)
        df['test_response_4_match_percentage'] = pd.Series(new_test_response_4_match_percentage)

        # One-hot encoding of discrete variables, including booleans (be sure to remove old features)

        categorical_columns = [
            'untagged_controls',
            'untagged_response',
            'passed_liveness',
            'connect_error',
            'in_control_group',
            'excluded',
            'excluded_is_CDN',
            'excluded_below_threshold',
            'control_response_start_success',
            'control_response_end_success',
            'control_response_start_has_type_a',
            'control_response_start_rcode',
            'control_response_end_has_type_a',
            'control_response_end_rcode',
            'test_query_successful',
            'test_noresponse_1_has_type_a',
            'test_noresponse_1_rcode',
            'test_noresponse_2_has_type_a',
            'test_noresponse_2_rcode',
            'test_noresponse_3_has_type_a',
            'test_noresponse_3_rcode',
            'test_noresponse_4_has_type_a',
            'test_noresponse_4_rcode',
            'test_response_has_type_a',
            'test_response_rcode',
            'more_IPs',
            'test_response_0_IP_match',
            'test_response_0_http_match',
            'test_response_0_cert_match',
            'test_response_0_asnum_match',
            'test_response_0_asname_match',
            'include_IP_0',
            'test_response_1_IP_match',
            'test_response_1_http_match',
            'test_response_1_cert_match',
            'test_response_1_asnum_match',
            'test_response_1_asname_match',
            'include_IP_1',
            'test_response_2_IP_match',
            'test_response_2_http_match',
            'test_response_2_cert_match',
            'test_response_2_asnum_match',
            'test_response_2_asname_match',
            'include_IP_2',
            'test_response_3_IP_match',
            'test_response_3_http_match',
            'test_response_3_cert_match',
            'test_response_3_asnum_match',
            'test_response_3_asname_match',
            'include_IP_3',
            'test_response_4_IP_match',
            'test_response_4_http_match',
            'test_response_4_cert_match',
            'test_response_4_asnum_match',
            'test_response_4_asname_match',
            'include_IP_4',
        ]

        for column in categorical_columns:
            temp_df = pd.get_dummies(df[column], prefix=column)

            df = pd.merge(left=df, right=temp_df, left_index=True, right_index=True)

            df = df.drop(columns=column)

        # Standard scale all variables
        # Note that all series metadata is lost at this point

        column_list = df.columns.values.tolist()

        numpy_df = df.to_numpy()

        scaler = StandardScaler()

        scaler = scaler.fit(numpy_df)

        with open(standard_scaler_file_name, "wb") as file:
            pickle.dump(scaler, file)

        numpy_df = scaler.transform(numpy_df)

        df = pd.DataFrame(numpy_df, columns=column_list)

        # Save file in parquet

        df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=(dataframe_ML_file_name +country+".gzip"))

    # Delete all members of folder temp_working_folder_unzipped_files (purpose: delete results.json after extraction complete)
    for file_name in os.listdir(temp_file_name):
        os.remove(temp_file_name+file_name)













