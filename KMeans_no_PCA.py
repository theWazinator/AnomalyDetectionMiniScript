import os.path
import pandas as pd
import datetime as dt
from methods import *
import random
import copy
import pickle
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from joblib import dump, load

input_path_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ReadyToGoDataset.gzip"
output_path_model_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/KMeans_no_PCA/V1_Model"
output_path_stats_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/KMeans_no_PCA/V1_Stats"
output_path_anomaly_vector = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/AnomalyVector"

# Input the data
df = pd.read_parquet(path=input_path_base, engine='pyarrow')

with open(output_path_anomaly_vector + ".txt", "rb") as file:
    anomaly_vector = pickle.load(file)

# Split the data (including the anomaly outputs) TODO change this split to 0.6 after testing
numpy_df = df.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(numpy_df, anomaly_vector, train_size=0.01, shuffle=False)

print("Split complete")

# Model parameters TODO need to try multiple times with different random seeds or other variables

clusters = 2
verbose = 1
batch_size = 5120
random_state = 123
reassignment_ratio = 0.015

#  Train the model
kmeans = MiniBatchKMeans(n_clusters=clusters,
                         verbose=verbose,
                         batch_size=batch_size,
                         random_state=random_state,
                         reassignment_ratio=reassignment_ratio)

kmeans.fit(X_train, y_train)

print("Training complete")

# Save the model

with open(output_path_model_base + ".joblib", "wb") as file:
     dump(kmeans, file)

print("Saving complete")

# TODO try prediction labels

test_results = kmeans.predict(X_test)

# Count

# TODO Compute summary statistics and save to file

# TODO Create tables of incorrectly-labelled samples and save to file

# TODO Save summary statistics and incorrectly-labelled samples to file

print(kmeans) # TODO remove this

print("All modules complete")