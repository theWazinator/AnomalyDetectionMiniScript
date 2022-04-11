import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_ML_features(df):

    columns_to_keep = [
        'average_matchrate',
        'untagged_controls',
        'untagged_response',
        'passed_liveness',
        'connect_error',
        'in_control_group',
        'excluded_below_threshold',
        'delta_time',
        'control_response_start_success',
        'control_response_end_success',
        'control_response_start_has_type_a',
        'control_response_start_rcode',
        'control_response_end_has_type_a',
        'control_response_end_rcode',
        'test_query_successful',
        'test_query_unsuccessful_attempts',
        'test_noresponse_1_has_type_a',
        'test_noresponse_1_rcode',
        'test_noresponse_2_has_type_a',
        'test_noresponse_2_rcode',
        'test_noresponse_3_has_type_a',
        'test_noresponse_3_rcode',
        'test_noresponse_4_has_type_a',
        'test_noresponse_4_rcode',
        'test_response_has_type_a',
        'test_response_rcode',
        'test_response_IP_count',
        'test_response_0_IP_match',
        'test_response_0_http_match',
        'test_response_0_cert_match',
        'test_response_0_asnum_match',
        'test_response_0_asname_match',
        'test_response_0_match_percentage',
        'test_response_0_asnum',
        'test_response_1_IP_match',
        'test_response_1_http_match',
        'test_response_1_cert_match',
        'test_response_1_asnum_match',
        'test_response_1_asname_match',
        'test_response_1_match_percentage',
        'test_response_1_asnum',
        'test_response_2_IP_match',
        'test_response_2_http_match',
        'test_response_2_cert_match',
        'test_response_2_asnum_match',
        'test_response_2_asname_match',
        'test_response_2_match_percentage',
        'test_response_2_asnum',
        'test_response_3_IP_match',
        'test_response_3_http_match',
        'test_response_3_cert_match',
        'test_response_3_asnum_match',
        'test_response_3_asname_match',
        'test_response_3_match_percentage',
        'test_response_3_asnum',
        'test_response_4_IP_match',
        'test_response_4_http_match',
        'test_response_4_cert_match',
        'test_response_4_asnum_match',
        'test_response_4_asname_match',
        'test_response_4_match_percentage',
        'test_response_4_asnum',
    ]

    df = df[df.columns.intersection(columns_to_keep)] # drop unwanted columns

    # Create new boolean columns to indicate more IPs
    row_count = df.shape[0]

    ip_count = df['test_response_IP_count']

    more_ips = np.full(shape=row_count, dtype=bool, fill_value=False)

    for index in range(0, row_count):
        if ip_count.iloc[index] > 5:
            more_ips[index] = True

    df['more_IPs'] = pd.Series(more_ips)

    #  Create new boolean columns to indicate if IP is being used

    include_IP_0 = np.full(shape=row_count, dtype=bool, fill_value=False)
    include_IP_1 = np.full(shape=row_count, dtype=bool, fill_value=False)
    include_IP_2 = np.full(shape=row_count, dtype=bool, fill_value=False)
    include_IP_3 = np.full(shape=row_count, dtype=bool, fill_value=False)
    include_IP_4 = np.full(shape=row_count, dtype=bool, fill_value=False)

    for index in range(0, row_count):
        if ip_count.iloc[index] == 1:
            include_IP_0[index] = True

        if ip_count.iloc[index] == 2:
            include_IP_1[index] = True

        if ip_count.iloc[index] == 3:
            include_IP_2[index] = True

        if ip_count.iloc[index] == 4:
            include_IP_3[index] = True

        if ip_count.iloc[index] == 5:
            include_IP_4[index] = True

    df['include_IP_0'] = pd.Series(include_IP_0)
    df['include_IP_1'] = pd.Series(include_IP_1)
    df['include_IP_2'] = pd.Series(include_IP_2)
    df['include_IP_3'] = pd.Series(include_IP_3)
    df['include_IP_4'] = pd.Series(include_IP_4)

    # Change IP match percentage rate to same as average if IP is not being used

    new_test_response_0_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)
    new_test_response_1_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)
    new_test_response_2_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)
    new_test_response_3_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)
    new_test_response_4_match_percentage = np.full(shape=row_count, dtype=np.float64(), fill_value=0)

    old_test_response_0_match_percentage = df['test_response_0_match_percentage']
    old_test_response_1_match_percentage = df['test_response_1_match_percentage']
    old_test_response_2_match_percentage = df['test_response_2_match_percentage']
    old_test_response_3_match_percentage = df['test_response_3_match_percentage']
    old_test_response_4_match_percentage = df['test_response_4_match_percentage']

    average_matchrate = df['average_matchrate']

    for index in range(0, row_count):
        if old_test_response_0_match_percentage.iloc[index] < 0:  # if empty IP, should be -2 in float notation if
            new_test_response_0_match_percentage[index] = average_matchrate[index]

        if old_test_response_1_match_percentage.iloc[index] < 0:  # if empty IP, should be -2 in float notation if
            new_test_response_1_match_percentage[index] = average_matchrate[index]

        if old_test_response_2_match_percentage.iloc[index] < 0:  # if empty IP, should be -2 in float notation if
            new_test_response_2_match_percentage[index] = average_matchrate[index]

        if old_test_response_3_match_percentage.iloc[index] < 0:  # if empty IP, should be -2 in float notation if
            new_test_response_3_match_percentage[index] = average_matchrate[index]

        if old_test_response_4_match_percentage.iloc[index] < 0:  # if empty IP, should be -2 in float notation if
            new_test_response_4_match_percentage[index] = average_matchrate[index]

    df['test_response_0_match_percentage'] = pd.Series(new_test_response_0_match_percentage)
    df['test_response_1_match_percentage'] = pd.Series(new_test_response_1_match_percentage)
    df['test_response_2_match_percentage'] = pd.Series(new_test_response_2_match_percentage)
    df['test_response_3_match_percentage'] = pd.Series(new_test_response_3_match_percentage)
    df['test_response_4_match_percentage'] = pd.Series(new_test_response_4_match_percentage)

    # One-hot encoding of discrete variables, including booleans (be sure to remove old features)

    categorical_columns = [
        'untagged_controls',
        'untagged_response',
        'passed_liveness',
        'connect_error',
        'in_control_group',
        'excluded_below_threshold',
        'control_response_start_success',
        'control_response_end_success',
        'control_response_start_has_type_a',
        'control_response_start_rcode',
        'control_response_end_has_type_a',
        'control_response_end_rcode',
        'test_query_successful',
        'test_noresponse_1_has_type_a',
        'test_noresponse_1_rcode',
        'test_noresponse_2_has_type_a',
        'test_noresponse_2_rcode',
        'test_noresponse_3_has_type_a',
        'test_noresponse_3_rcode',
        'test_noresponse_4_has_type_a',
        'test_noresponse_4_rcode',
        'test_response_has_type_a',
        'test_response_rcode',
        'more_IPs',
        'test_response_0_IP_match',
        'test_response_0_http_match',
        'test_response_0_cert_match',
        'test_response_0_asnum_match',
        'test_response_0_asname_match',
        'test_response_0_asnum',
        'include_IP_0',
        'test_response_1_IP_match',
        'test_response_1_http_match',
        'test_response_1_cert_match',
        'test_response_1_asnum_match',
        'test_response_1_asname_match',
        'test_response_1_asnum',
        'include_IP_1',
        'test_response_2_IP_match',
        'test_response_2_http_match',
        'test_response_2_cert_match',
        'test_response_2_asnum_match',
        'test_response_2_asname_match',
        'test_response_2_asnum',
        'include_IP_2',
        'test_response_3_IP_match',
        'test_response_3_http_match',
        'test_response_3_cert_match',
        'test_response_3_asnum_match',
        'test_response_3_asname_match',
        'test_response_3_asnum',
        'include_IP_3',
        'test_response_4_IP_match',
        'test_response_4_http_match',
        'test_response_4_cert_match',
        'test_response_4_asnum_match',
        'test_response_4_asname_match',
        'test_response_4_asnum',
        'include_IP_4',
    ]

    for column in categorical_columns:
        temp_df = pd.get_dummies(df[column], prefix=column)

        df = pd.merge(left=df, right=temp_df, left_index=True, right_index=True)

        df = df.drop(columns=column)

    # Standard scale all variables
    # Note that all series metadata is lost at this point

    column_list = df.columns.values.tolist()

    numpy_df = df.to_numpy()

    scaler = StandardScaler()

    scaler = scaler.fit(numpy_df)

    numpy_df = scaler.transform(numpy_df)

    df = pd.DataFrame(numpy_df, columns=column_list)

    return df