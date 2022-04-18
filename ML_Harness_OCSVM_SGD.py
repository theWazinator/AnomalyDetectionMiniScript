"""
This is the harness that runs the ML models.
Here we run linear_model.SGDOneClassSVM, a linear-approximation of the OCSVM
 """

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDOneClassSVM
import time
from itertools import islice
from multiprocessing import Manager, Process
import os
from joblib import dump, load


model_name = "OCSVM_SGD_skl"
version = 1
version_filename = r"/home/jambrown/CP_Analysis/ML_Results/OCSVM_SGD/V" +str(version)+ "/"
sklearn_bool = True
model_set_list = [1,2,3,4]
os.mkdir(version_filename)

params_1 = {'nu': 0.5,
            'fit_intercept': True,
            'max_iter': 1000,
            'tol': 0.001,
            'shuffle': True,
            'verbose': 1,
            'random_state': 23452345,
            'learning_rate': 'optimal',
            'eta0': 0.0,
            'power_t': 0.5,
            'warm_start': False,
            'average': False,
}

params_2 = {'nu': 0.5,
            'fit_intercept': True,
            'max_iter': 1000,
            'tol': 0.001,
            'shuffle': True,
            'verbose': 1,
            'random_state': 23452345,
            'learning_rate': 'optimal',
            'eta0': 0.0,
            'power_t': 0.5,
            'warm_start': False,
            'average': False,
}

params_3 = {'nu': 0.5,
            'fit_intercept': True,
            'max_iter': 1000,
            'tol': 0.001,
            'shuffle': True,
            'verbose': 1,
            'random_state': 23452345,
            'learning_rate': 'optimal',
            'eta0': 0.0,
            'power_t': 0.5,
            'warm_start': False,
            'average': False,
}

params_4 = {'nu': 0.5,
            'fit_intercept': True,
            'max_iter': 1000,
            'tol': 0.001,
            'shuffle': True,
            'verbose': 1,
            'random_state': 23452345,
            'learning_rate': 'optimal',
            'eta0': 0.0,
            'power_t': 0.5,
            'warm_start': False,
            'average': False,
}

def get_results(validation_set_df, validation_truth_df, anomaly_df, model, model_params, save_folder, t_num, time_elapsed):

    predicted_results_list = model.predict(validation_set_df.to_numpy())
    true_results_list = validation_truth_df.to_numpy()
    anomaly_results_list = anomaly_df.to_numpy()

    if sklearn_bool == True:

        # If the model comes from scikit-learn, we need to convert the feature indicator into 0 for inlier and 1 for outlier
        predicted_results_list = convert_target_features(predicted_results_list)

    # TODO include saving of AUC curves

    # TODO compare to results from 'anomaly' from same dataset

    return results_df

def run_ml_model(training_set_df, validation_set_df, validation_truth_df, anomaly_df, model_params, save_folder, t_num):

    print("Begin training model T = " +str(t_num))
    begin_time = time.time()
    clf = SGDOneClassSVM(**model_params)
    clf.fit(training_set_df)
    time_elapsed = time.time() - begin_time
    print("End training model T= " +str(t_num))
    print("Time Elapsed: " +str(time_elapsed)+ " seconds.")
    dump(clf, save_folder +'model.joblib')

    ("Begin validating results for T= " + str(t_num))

    results_df = get_results(validation_set_df, validation_truth_df, anomaly_df, clf, model_params, save_folder, t_num, time_elapsed)

    results_df.to_csv(save_folder +r"local_stats.csv", index=False)

    ("Finished validating and saving results for T= " +str(t_num))

# TODO include timing of ML-training and all other variables in spreadsheet

print("Begin machine learning harness")

home_file_name = r"/home/jambrown/CP_Analysis/"
ps = list()
for model_set in model_set_list:

    if model_set == 1:

        country_code = "CN"
        country_name = "China"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Clean_descriptiveFeatures_fullDataset.gzip'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Clean_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name +r'VALIDATION_Clean_targetFeature_GFWatch_Censored.csv'
        anomaly_file_name = ml_ready_data_file_name +r'VALIDATION_Clean_targetFeature_anomaly.csv'

        model_params = params_1

    elif model_set == 2:

        country_code = "CN"
        country_name = "China"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Clean_descriptiveFeatures_fullDataset.gzip'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_targetFeature_GFWatch_Censored.csv'
        anomaly_file_name = ml_ready_data_file_name + r'VALIDATION_Mixed_targetFeature_anomaly.csv'

        model_params = params_2

    elif model_set == 3:

        country_code = "US"
        country_name = "United States"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Clean_descriptiveFeatures_fullDataset.gzip'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Clean_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name +r'VALIDATION_Clean_targetFeature_Presumed_Censored.csv'
        anomaly_file_name = ml_ready_data_file_name + r'VALIDATION_Clean_targetFeature_anomaly.csv'

        model_params = params_3

    elif model_set == 4:

        country_code = "US"
        country_name = "United States"

        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes/"

        training_set_file_name = ml_ready_data_file_name +r'TRAINING_Clean_descriptiveFeatures_fullDataset.gzip'
        validation_set_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_descriptiveFeatures_fullDataset.gzip'
        validation_truth_file_name = ml_ready_data_file_name +r'VALIDATION_Mixed_targetFeature_Presumed_Censored.csv'
        anomaly_file_name = ml_ready_data_file_name + r'VALIDATION_Mixed_targetFeature_anomaly.csv'

        model_params = params_4

    training_set_df = pd.read_parquet(path=training_set_file_name,
        engine='pyarrow').iloc[0:5000] #TODO After debugging, remove selective indices


    validation_set_df = pd.read_parquet(path=validation_set_df,
        engine='pyarrow')

    validation_truth_df = pd.read_csv(path=validation_truth_file_name)

    anomaly_df = pd.read_csv(path=anomaly_file_name)

    save_folder = version_filename + r"T" +str(model_set) + r"/"

    p = Process(target=run_ml_model,
                args=(training_set_df, validation_set_df, validation_truth_df, anomaly_df, model_params, save_folder, model_set))
    ps.append(p)
    p.start()

for p in ps:
    p.join()

# Create csv containing all pertinent information about the models, their parameters, and the results
partial_df_list = []

for model_set in model_set_list:

    model_file_name = version_filename + r"T" +str(model_set) + r"/local_stats.csv"

    partial_df = pd.read_csv(path=model_file_name)

    partial_df_list.append(partial_df)

master_df = pd.concat(partial_df_list, ignore_index=True, axis=0)

# Relabel Records
master_df.reset_index(drop=True, inplace=True)

#Save to file
master_df.to_csv(index=False, path=version_filename + r"statistics.csv")
print("End of machine learning harness for Version " +str(version) +".", flush=True)



