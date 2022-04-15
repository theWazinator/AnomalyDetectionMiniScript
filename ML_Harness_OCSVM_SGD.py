"""
This is the harness that runs the ML models.
Here we run linear_model.SGDOneClassSVM, a linear-approximation of the OCSVM
 """

import matplotlib.pyplot as plt
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
os.mkdir(version_filename)

params_1 = {'nu': 0.5,
            'fit_intercept': True,
            'max_iter': 1000,
            'tol': 0.001,
            'shuffle': True,
            'verbose': 0,
            'random_state': None,
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
            'verbose': 0,
            'random_state': None,
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
            'verbose': 0,
            'random_state': None,
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
            'verbose': 0,
            'random_state': None,
            'learning_rate': 'optimal',
            'eta0': 0.0,
            'power_t': 0.5,
            'warm_start': False,
            'average': False,
}

def dict_to_key_value_list(old_dt):

    new_dt ={}

    for key in old_dt.keys():
        new_dt[key] = [old_dt[key]]

    return new_dt

def save_to_gzip_and_print_report():

# This section should access the saved model in order to be compatible for using the test set
def get_results(trial_set_df, trial_truth_list, model, model_params, model_library, save_folder_name)

def run_ml_model(training_set_df, validation_set_df, validation_truth_list, model_params, model_library):

# TODO remember that validation_truth_list is written with 1 for outliers and 0 for inliers - this is the format for PyOd
# TODO in sci-kit learn, the values are 1 for inliers and -1 for outliers

# TODO include timing of ML-training and all other variables in spreadsheet

# TODO include saving of AUC curves

ps = list()

for model_set in [1, 2, 3, 4]:

    if model_set == 1:

        country_code = "CN"
        country_name = "China"

        home_file_name = r"/home/jambrown/CP_Analysis/"
        ml_ready_data_file_name = home_file_name + country_code + "/ML_ready_dataframes/"

        training_set_file_name =
        validation_set_file_name =
        validation_truth_file_name =

    elif model_set == 2:

    elif model_set == 3:

    elif model_set == 4:


    # TODO convert training_set, validation_set into dataframes and validation_truth into a list

    p = Process(target=run_ml_model,
                args=(training_set_df, validation_set_df, validation_truth_list, model_params, model_set))
    ps.append(p)
    p.start()

for p in ps:
    p.join()

# TODO Create joint csv for copying and pasting


