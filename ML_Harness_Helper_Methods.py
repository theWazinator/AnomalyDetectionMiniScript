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