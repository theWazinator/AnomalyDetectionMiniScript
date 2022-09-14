import pandas as pd
import numpy as np

def round_list(old_list):

    new_list = []

    for index in range(0, len(old_list)):

        new_list.append(round(old_list[index], 0))

    return new_list

def dict_to_key_value_list(old_dt):

    new_dt ={}

    for key in old_dt.keys():
        new_dt[key] = [old_dt[key]]

    return new_dt

# In sci-kit learn, the values are 1 for inliers and -1 for outliers.
# This method converts the list to 0 for inliers and 1 for outliers
def convert_target_features(input_list):

    output_list = []

    for index in range(0, len(input_list)):

        if input_list[index] == 1:
            output_list.append(0)

        elif input_list[index] == -1:
            output_list.append(1)

    return output_list

def calculate_counts(prediction_list, truth_list):

    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    for index in range(0, len(prediction_list)):

        if prediction_list[index] == 0 and truth_list[index] == 0:

            tn_count += 1

        elif prediction_list[index] == 0 and truth_list[index] == 1:

            fn_count += 1

        elif prediction_list[index] == 1 and truth_list[index] == 0:

            fp_count += 1

        elif prediction_list[index] == 1 and truth_list[index] == 1:

            tp_count += 1

    assert(sum([tp_count, tn_count, fp_count, fn_count]) == len(prediction_list))

    return tp_count, tn_count, fp_count, fn_count

# TODO TEST complete this function
def remove_features(training_set_df, validation_set_df, validation_truth_df, validation_comparison_df, features_to_remove):

    df_list = [training_set_df, validation_set_df, validation_truth_df, validation_comparison_df]

    for index in range(0, len(df_list)):

        complex_feature_list = df_list[index].columns.tolist()  # used later to group features for feature importance

        complex_features_to_remove = []

        for complex_feature in complex_feature_list:

            for feature_to_remove in features_to_remove:

                if feature_to_remove in complex_feature:

                    # TODO TEST Logic for ensuring test_response_i_asnum and test_response_i_asnum_match are not confused
                    if not ("asnum_match" not in feature_to_remove and "asnum_match" in complex_feature):

                        complex_features_to_remove.append(complex_feature)

        df_list[index].drop(complex_features_to_remove, axis=1, inplace=True)

    # remove the features from the df
    print("Finished removing features")

    return df_list[0].copy(), df_list[1].copy(), df_list[2].copy(), df_list[3].copy()

# TODO TEST aggregate feature importances

def aggregate_feature_importances(base_feature_list, sorted_features_list, sorted_num_list):

    base_feature_dict = {}

    for base_feature in base_feature_list:

        base_feature_dict[base_feature] = 0

    index = 0
    for complex_feature in sorted_features_list:

        feature_weight = sorted_num_list[index]

        for base_feature in base_feature_list:

            if base_feature in complex_feature:

                # TODO TEST Logic for ensuring test_response_i_asnum and test_response_i_asnum_match are not confused
                if not ("asnum_match" not in base_feature and "asnum_match" in complex_feature):
                    base_feature_dict[base_feature] = base_feature_dict[base_feature] + feature_weight

        index += 1

    base_feature_df_list = []
    aggregate_weight_df_list = []

    for base_feature in base_feature_dict.keys():
        base_feature_df_list.append(base_feature)
        aggregate_weight_df_list.append(base_feature_dict[base_feature])

    base_feature_weight_dict = {"Features": base_feature_df_list, "Importance": aggregate_weight_df_list}

    base_feature_weight_df = pd.DataFrame.from_dict(base_feature_weight_dict)

    return base_feature_weight_df



