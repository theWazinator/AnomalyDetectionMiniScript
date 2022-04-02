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

# Create a text document of IP addresses in Team Cyru format that can be used to find AS information about the VPs

# Get list of files in folder
cp_downloads_zipped_file_name = r"/home/jambrown/CP_Downloads/"
home_file_name = r"/home/jambrown/CP_Analysis/"

countries_to_select = ["United States", "China", "India", "Russia", "Turkey", "Iran", "Turkmenistan"]

list_of_zipped_cp_files = os.listdir(cp_downloads_zipped_file_name)
list_of_zipped_cp_files.sort(reverse=True) # Ensures most recent scans are processed first

# TODO add parallel lis counting how many times each appears

VP_dict = {}

for zipped_cp_file in list_of_zipped_cp_files:

    cp_scan_only_name = zipped_cp_file.split('.')[0]

    resolvers_filename = r'/home/jambrown/CP_Analysis/' +cp_scan_only_name+  r'/other_docs/resolvers.json'

    reader = open(resolvers_filename, 'r')

    try:

        for line in reader:
            data = json.loads(line)  # Convert json to dictionary

            if data['location']['country_name'] in countries_to_select:

                if data['vp'] not in VP_dict.keys():

                    VP_dict[data['vp']] = 1

                else:

                    VP_dict[data['vp']] = VP_dict[data['vp']] + 1

    finally:
        reader.close()

VP_list = []
VP_count = []

# Print to file

print('begin')

for key in VP_dict.keys():

    print(key)

    VP_list.append(key)
    VP_count.append(VP_dict[key])

print('end')

# Save to CSV file

VP_count_dict = {

    "Vantage Point": VP_list,
    "Count": VP_count,
}

df = pd.DataFrame(VP_count_dict)

df.to_csv("VantagePoint_Count.csv")