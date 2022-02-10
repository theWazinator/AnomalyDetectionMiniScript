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
from sklearn.decomposition import PCA
from joblib import dump, load

input_path_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ReadyToGoDataset.gzip"
input_path_anomaly_vector = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/AnomalyVector"
records_file_complete = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/RecordsSelected.gzip"

# Changes with every model?
model_folder_name = 'KMeans_with_PCA/V1'
train_size = 0.6 #TODO change this split to 0.6 after testing
# Model parameters
clusters = 2
verbose = 1
batch_size = 5120
random_state = 123

# Set up the model
pca = PCA(random_state=random_state)

kmeans = MiniBatchKMeans(n_clusters=clusters,
                         verbose=verbose,
                         batch_size=batch_size,
                         random_state=random_state)

output_path_model_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_Model"
output_path_stats_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_Stats"
output_path_pca_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_PCA"
output_path_records_to_keep_no_censorship_fp = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_no_censorship_fp.gzip"
output_path_records_to_keep_yes_censorship_fn = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_yes_censorship_fn.gzip"
output_path_records_to_keep_yes_censorship_maybe_positive = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_yes_censorship_maybe_positive.gzip"
output_path_records_to_keep_yes_censorship_maybe_negative = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_yes_censorship_maybe_negative.gzip"

# Input the data
df = pd.read_parquet(path=input_path_base, engine='pyarrow')

complete_records_df = pd.read_parquet(path=records_file_complete, engine='pyarrow')
country_name = complete_records_df["country_name"]

with open(input_path_anomaly_vector + ".txt", "rb") as file:
    anomaly_vector = pickle.load(file)

print("Data Input Complete")

# Split the data (including the anomaly outputs)
X_train, X_test, anomaly_train, anomaly_test = train_test_split(df, anomaly_vector, train_size=train_size, shuffle=False)
_, country_name_test = train_test_split(country_name, train_size=train_size, shuffle=False)
country_name_test = country_name_test.to_numpy()

print("Split complete")

# PCA preprocessing
pca.fit(X_train)
X_train = pca.transform(X_train)

with open(output_path_pca_base + ".joblib", "wb") as file:
    dump(pca, file)

print("PCA Preprocessing Complete")

#  Train the KMeans model

kmeans.fit(X_train, anomaly_train)

print("Training Complete")

# Save the model

with open(output_path_model_base + ".joblib", "wb") as file:
     dump(kmeans, file)

print("Saving Complete")

# Predict results and determine cluster labels

test_predictions = kmeans.predict(X_test)

# Count number of each cluster and set lower one for each
# Changes with every model

count_0s = 0
count_1s = 1

for i in range(0, len(test_predictions)):

    if test_predictions[i] == 0:
        count_0s += 1
    else:
        count_1s += 1

if count_0s > count_1s:
    inlier_group_num = 0
    outlier_group_num = 1
    inlier_group_points = count_0s
    outlier_group_points = count_1s

else:
    inlier_group_num = 1
    outlier_group_num = 0
    inlier_group_points = count_1s
    outlier_group_points = count_0s

print("Prediction Complete")

# Compute summary statistics and save to file

tp_count = 0 # True positive count
fp_count = 0 # False positive count
tn_count = 0 # True negative count
fn_count = 0 # False negative count
maybe_p_count = 0 # Positives for censored countries that are marked as non-anomalous by Censored Planet
maybe_n_count = 0
testSamples_count = len(test_predictions)

records_to_keep_no_censorship_fp = [] # List of indices of records to keep
records_to_keep_yes_censorship_fn = [] # List of indices of records to keep
records_to_keep_yes_censorship_maybe_positive = [] # List of indices of records to keep
records_to_keep_yes_censorship_maybe_negative = [] # List of indices of records to keep

total_included_samples_count = 0 # The number of samples used in the denominator for threshold calculation
offset = len(X_train) # offset to add to indices for records_to_keep to ensure that it aligns with original indices of the records

for i in range(0, testSamples_count):

    if country_name_test[i] in ["United States", "Canada"]:

        total_included_samples_count += 1

        # There are no positive anomalies in the original dataset
        if test_predictions[i] == outlier_group_num:
            fp_count += 1
            records_to_keep_no_censorship_fp.append(i+offset)
        else:
            tn_count += 1

    else:
        if test_predictions[i] == outlier_group_num:
            if anomaly_test[i] == True:
                tp_count += 1
                total_included_samples_count += 1
            else:
                maybe_p_count += 1
                records_to_keep_yes_censorship_maybe_positive.append(i+offset)
        else:
            if anomaly_test[i] == True:
                fn_count += 1
                records_to_keep_yes_censorship_fn.append(i+offset)
                total_included_samples_count += 1
            else:
                maybe_n_count += 1
                records_to_keep_yes_censorship_maybe_negative.append(i+offset)

accuracy = (tn_count + tp_count)/total_included_samples_count

# TODO remove
assert(total_included_samples_count == tp_count + fp_count + tn_count + fn_count)

fpr = fp_count / (fp_count + tn_count)
tpr = tp_count / (tp_count + fn_count)
fnr = fn_count / (fn_count + tp_count)
tnr = tn_count / (tn_count + fp_count)
f1 = (2*tp_count)/(2*tp_count+fp_count+fn_count)

with open(output_path_stats_base + ".txt", 'w') as file:
    file.write('Accuracy: ' + str(accuracy))
    file.write('\nTrue Positives: ' + str(tp_count))
    file.write('\nTrue Negatives: ' + str(tn_count))
    file.write('\nFalse Positives: ' + str(fp_count))
    file.write('\nFalse Negatives: ' + str(fn_count))
    file.write('\nSensitivity (TPR): ' + str(tpr))
    file.write('\nSpecificity (TNR): ' + str(tnr))
    file.write('\nFalse Positive Rate: ' + str(fpr))
    file.write('\nFalse Negative Rate: ' + str(fnr))
    file.write('\nF1 Score: ' + str(f1))

    file.write('\n\nPossible Outliers Not Identified by CP: ' + str(maybe_p_count))
    file.write('\nPossible Inliers Not Identified by CP: ' + str(maybe_n_count))

    # Changes with every model
    file.write('\n\nInlier Cluster Number: ' + str(inlier_group_num))
    file.write('\nOutlier Cluster Number: ' + str(outlier_group_num))
    file.write('\n\nInlier Cluster Count: ' + str(inlier_group_points))
    file.write('\nOutlier Cluster Count: ' + str(outlier_group_points))

    # Changes with every model
    file.write("\n\nModel Parameters: ")
    file.write("\nTraining Fraction: " + str(train_size))
    file.write('\nClusters: ' + str(clusters))
    file.write('\nVerbose: ' + str(verbose))
    file.write('\nBatch Size: ' + str(batch_size))
    file.write('\nRandom State: ' + str(random_state))

    # Changes with every model
    file.write('\n\nModel training output: ')
    file.write('\nInertia: ' + str(kmeans.inertia_))
    file.write('\nIteration Count: ' + str(kmeans.n_iter_))
    file.write('\nMinibatch Count: ' + str(kmeans.n_steps_))

    # Changes with every model
    file.write("\n\nPCA Parameters: ")
    file.write("\nComponent Count: " + str(pca.n_components))
    file.write("\nExplained Variance: ")
    file.write(str(pca.explained_variance_ratio_))

print("Summary Statistics Complete")

no_censorship_fp_df = complete_records_df[complete_records_df.index.isin(records_to_keep_no_censorship_fp) == True]
yes_censorship_fn_df = complete_records_df[complete_records_df.index.isin(records_to_keep_yes_censorship_fn) == True]
yes_censorship_maybe_positive_df = complete_records_df[complete_records_df.index.isin(records_to_keep_yes_censorship_maybe_positive) == True]
yes_censorship_maybe_negative_df = complete_records_df[complete_records_df.index.isin(records_to_keep_yes_censorship_maybe_negative) == True]

no_censorship_fp_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=output_path_records_to_keep_no_censorship_fp)
print(no_censorship_fp_df.info())
yes_censorship_fn_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=output_path_records_to_keep_yes_censorship_fn)
print(yes_censorship_fn_df.info())
yes_censorship_maybe_positive_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=output_path_records_to_keep_yes_censorship_maybe_positive)
print(yes_censorship_maybe_positive_df.info())
yes_censorship_maybe_negative_df.to_parquet(index=True, compression="gzip", engine='pyarrow',  path=output_path_records_to_keep_yes_censorship_maybe_negative)
print(yes_censorship_maybe_negative_df.info())

print("Strange Predictions Table Complete")

print("Program Finished")