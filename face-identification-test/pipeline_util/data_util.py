import pandas as pd
import numpy as np
import torch


def panda_testing_entries_from_context(context):
    # TODO Error Handling
    df = pd.DataFrame(context.testing_result_entries)
    is_actual_match_column = []
    rotation_angle_column = []
    expression_column = []
    lighting_column = []
    # TODO also the other columns
    for i, row in df.iterrows():
        is_actual_match_column.append(context.open_testing_entry[row['open_testing_entry_id']].is_actual_match)
        rotation_angle_column.append(context.open_testing_entry[row['open_testing_entry_id']].rotation_angle)
        expression_column.append(context.open_testing_entry[row['open_testing_entry_id']].expression)
        lighting_column.append(context.open_testing_entry[row['open_testing_entry_id']].lighting)

    df['is_actual_match'] = is_actual_match_column
    df['rotation_angle'] = rotation_angle_column
    df['expression'] = expression_column
    df['lighting'] = lighting_column
    df['decision'] = np.nan  # Decision going to be made later on

    return df

def panda_failed_testing_entries_from_context(context):
    return pd.DataFrame(context.failed_testing_entries)

#
# Remap the deep3D Coeff dict to a 2D Array to be able to pass it to the model generator
#
def get_coeff_array_from_coeff_dict(deep_3d_coeff_dict):
    return torch.cat((
        deep_3d_coeff_dict['id'],
        deep_3d_coeff_dict['exp'],
        deep_3d_coeff_dict['tex'],
        deep_3d_coeff_dict['angle'],
        deep_3d_coeff_dict['gamma'],
        deep_3d_coeff_dict['trans'],
    ), dim=1)
