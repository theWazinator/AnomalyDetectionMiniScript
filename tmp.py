import os.path
import pandas as pd
import datetime as dt
from dataframe_processing_helper_methods import *
import random
import copy
import pickle

output_path_base = r"C:\Users\jacob\PycharmProjects\AnomalyDetectionMiniScript\TESTING_Clean_descriptiveFeatures_fullDataset.gzip"

df_tmp = pd.read_parquet(path=output_path_base, engine='pyarrow').iloc[0:1000]

#print to CSV
df_tmp.to_csv(path_or_buf=r"C:\Users\jacob\PycharmProjects\AnomalyDetectionMiniScript\TESTING_Clean_descriptiveFeatures_fullDataset.csv", \
             index=False)

print(df_tmp.columns.values.tolist())

print(df_tmp.info())
print(df_tmp.head(25))

# Get list of files in folder

# countries_to_select = ["United States", "China", "India", "Russia", "Turkey", "Iran"]
#
# print('begin')
#
# for zipped_cp_file in ['CP_Satellite-2021-10-17-12-00-01']:
#
#     cp_scan_only_name = zipped_cp_file.split('.')[0]
#
#     resolvers_filename = r'C:\Users\jacob\OneDrive\Documents\Princeton\Masters\Research\Data Downloads\Satellite-Iris\CP_Satellite-2021-10-17-12-00-01.tar\CP_Satellite-2021-10-17-12-00-01\CP_Satellite-2021-10-17-12-00-01\resolvers.json'
#
#     reader = open(resolvers_filename, 'r')
#
#     try:
#
#         for line in reader:
#             data = json.loads(line)  # Convert json to dictionary
#
#             if data['location']['country_name'] in countries_to_select:
#
#                 print(data['vp'])
#
#     finally:
#         reader.close()
#
# print('end')








