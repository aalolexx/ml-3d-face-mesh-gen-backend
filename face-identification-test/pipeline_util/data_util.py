import pandas as pd
import numpy as np
import torch

def get_panda_df_from_context(context):
    # TODO Error Handling
    df = pd.DataFrame(context.testing_result_entries)
    is_actual_match_column = []
    rotation_angle_column = []
    # TODO also the other columns
    for i, row in df.iterrows():
        is_actual_match_column.append(context.open_testing_entry[row['open_testing_entry_id']].is_actual_match)
        rotation_angle_column.append(context.open_testing_entry[row['open_testing_entry_id']].rotation_angle)

    df['is_actual_match'] = is_actual_match_column
    df['rotation_angle'] = rotation_angle_column
    df['decision'] = np.nan
    return df


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
