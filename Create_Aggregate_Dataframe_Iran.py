import os
import pandas as pd

cur_country_name = "Iran"
cur_country_code = "IR"
home_file_name = r"/home/jambrown/CP_Analysis/"
cp_downloads_zipped_file_name = r"/home/jambrown/CP_Downloads/"

# Create Country-specific file directory
country_file_name = home_file_name +cur_country_code+ r'/'
os.mkdir(country_file_name)

# Merge raw files into singular file for use
print("Begin merging raw dataframes into a singular dataframe.", flush=True)

list_of_zipped_cp_files = os.listdir(cp_downloads_zipped_file_name)
list_of_zipped_cp_files.sort(reverse=True) # Ensures most recent scans are processed first

print("Input Vantage Point Files")

vp_df = pd.read_csv("VantagePoint_CSV_List.csv")

vp_df = vp_df[vp_df.COUNTRY_CODE == cur_country_code]

vp_list = vp_df['IP'].tolist()

print("Begin merging " +cur_country_name+ " dataframes.", flush=True)

# Create dataframe with all the files
# Start by reading an initial dataframe to obtain the column headers
df_0 = pd.read_parquet(path='/home/jambrown/CP_Analysis/CP_Satellite-2022-02-09-12-00-01/raw_dataframes/0_raw_dataframe.gzip', engine='pyarrow')
master_df = pd.DataFrame.from_records(data=df_0, nrows=1)
master_df.drop(labels=master_df.index[0:], inplace=True)

# TODO note that we assume that the file names in the CP_Download folder are exactly the same as those in CP_Analysis
# TODO daily scan folders should probably be placed in a separate file so that there names can be processed properly
for zipped_cp_file in list_of_zipped_cp_files:

    cp_scan_only_name = zipped_cp_file.split('.')[0]
    print("Beginning to process file: " +str(cp_scan_only_name), flush=True)

    scan_base_file_name = home_file_name + cp_scan_only_name + r"/"
    other_docs_file_name = scan_base_file_name + r"other_docs/"
    raw_dataframes_file_name = scan_base_file_name + r'raw_dataframes/'

    splitfile_count = len(os.listdir(raw_dataframes_file_name))

    # Concatenate dataframes, selecting by country and vantage point
    for fileNumber in range(0, splitfile_count):

        partial_df = pd.read_parquet(path=raw_dataframes_file_name + str(fileNumber) + "_raw_dataframe.gzip", engine='pyarrow')

        # Select only the vantage points we want

        partial_df = partial_df[partial_df.vantage_point.isin([vp_list]) == True]

        # Merge with master dataframe
        master_df = pd.concat([master_df, partial_df], ignore_index=True)

        print("Finished merging file number " + str(fileNumber) +" of " +str(splitfile_count), flush=True)

    print("Finished merging the scan from day " +cp_scan_only_name)

# Shuffle records
master_df = master_df.sample(frac=1)

# Relabel Records
master_df.reset_index(drop=True, inplace=True)

# Save file
# fastparquet must be used so that the categorical variables associated with integers (ex rcode) will also deserialize into categorical variables
master_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=country_file_name + "raw_dataframe.gzip")

print("End of merging " + cur_country_name + " dataframes.", flush=True)

print("End of raw dataframe merging section.", flush=True)