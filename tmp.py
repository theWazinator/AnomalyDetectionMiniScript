import os.path
import pandas as pd
import datetime as dt
from methods import *
import random
import copy
import pickle

output_path_base = r"C:\Users\jacob\OneDrive\Documents\Princeton\Masters\Research\Isolation Forest Discovered Anomalies\PositiveAnomalies.gzip"

df_tmp = pd.read_parquet(path=output_path_base, engine='pyarrow')

#print to CSV
df_tmp.to_csv(path_or_buf=r"C:\Users\jacob\OneDrive\Documents\Princeton\Masters\Research\Isolation Forest Discovered Anomalies\PositiveAnomalies.csv", \
             index=False)

print(df_tmp.columns.values.tolist())

print(df_tmp.info())
print(df_tmp.head(25))









