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

countries_to_select = ["United States", "China", "India", "Russia", "Turkey", "Iran"]

list_of_zipped_cp_files = os.listdir(cp_downloads_zipped_file_name)
list_of_zipped_cp_files.sort(reverse=True) # Ensures most recent scans are processed first

print('begin')

for zipped_cp_file in list_of_zipped_cp_files:

    cp_scan_only_name = zipped_cp_file.split('.')[0]

    resolvers_filename = r'/home/jambrown/CP_Analysis/' +cp_scan_only_name+  r'/other_docs/resolvers.json'




print('end')