import os.path
import pandas as pd
import datetime as dt
from methods import *
import random
import copy
import pickle
from sklearn.model_selection import train_test_split
import hdbscan
from joblib import dump, load

input_path_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ReadyToGoDataset.gzip"
input_path_anomaly_vector = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/AnomalyVector"
records_file_complete = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/RecordsSelected.gzip"

# Changes with every model?
model_folder_name = 'HDBSCAN/V3'
train_size = 0.01 # TODO change this split to 0.6 after testing
# Model parameters
min_cluster_size = int(2724471*train_size*(1-0.01-0.0316465))  # 60% of the training set times (1% + 3.16465%) (Buffer % + Outlier %)
min_samples = 50
cluster_selection_epsilon = 0.5
core_dist_n_jobs = 10
allow_single_cluster = True
prediction_data = True
metric = 'euclidean'

print(model_folder_name)

# Set up the model
hdbscan_clust = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                          min_samples=min_samples,
                          cluster_selection_epsilon = cluster_selection_epsilon,
                          core_dist_n_jobs=core_dist_n_jobs,
                          allow_single_cluster = allow_single_cluster,
                          metric=metric,
                          prediction_data=prediction_data)

output_path_model_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_Model"
output_path_stats_base = r"/home/jambrown/CP_ML_Experiments/OneDay_Satellite/ModelOutput/" +model_folder_name+ r"_Stats"
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

#  Train the model

hdbscan_clust.fit_predict(X_train, anomaly_train)

print("Training Complete")

# Predict new points

test_predictions, strengths = hdbscan.prediction.approximate_predict(hdbscan_clust, X_test)

print("Prediction Complete")

# Save the model

with open(output_path_model_base + ".joblib", "wb") as file:
     dump(hdbscan_clust, file)

print("Saving Complete")

# Count number of each cluster and set lower one for each
# Changes with every model

count_outliers = 0
count_inliers = 0
cluster_list = []

for i in range(0, len(test_predictions)):

    if test_predictions[i] == -1:
        count_outliers += 1
    else:
        count_inliers += 1

        if test_predictions[i] not in cluster_list:
            cluster_list.append(test_predictions)

outlier_group_num = -1

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
    file.write('\nOutlier Cluster Number: ' + str(outlier_group_num))
    file.write('\n\nInlier Cluster Count: ' + str(len(cluster_list)))
    file.write('\nOutlier Count: ' + str(count_outliers))
    file.write('\nInlier Count: ' + str(count_inliers))

    min_cluster_size = 1566603  # 60% of the training set times (1% + 3.16465%) (Buffer % + Outlier %)
    min_samples = 50
    cluster_selection_epsilon = 10
    allow_single_cluster = True
    metric = 'euclidean'

    # Changes with every model
    file.write("\n\nModel Parameters: ")
    file.write("\nMin Cluster Size: " + str(min_cluster_size))
    file.write('\nMin Samples: ' + str(min_samples))
    file.write('\nCluster Selection Epsilon: ' + str(cluster_selection_epsilon))
    file.write('\nNumber of Jobs (CPUs): ' + str(core_dist_n_jobs))
    file.write('\nAllow Single Cluster: ' + str(allow_single_cluster))
    file.write('\nMetric: ' + str(metric))

    # Changes with every model
    file.write('\n\nModel training output: ')
    file.write('\nCluster Persistence: ' + str(hdbscan_clust.cluster_persistence_))

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