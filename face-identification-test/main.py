import sys
sys.path.append('../deep-3d-face-recon')

from pipeline.pipeline import *
from pipeline_modules.context import Context
from pipeline_modules.data_preparation.data_preparation_3d import DataPreparation3D
from pipeline_modules.data_preparation.data_preparation_lfw import DataPreparationLFW
from pipeline_modules.deep_3d_coefficient_generator import Deep3DCoefficientGenerator

# Set up Context

ctx = Context(
    input_dir_path='data/input',
    output_dir_path='data/output',
    working_dir_path='data/working',
    misc_dir_path='data/misc',
    deep_3d_coeffs=None,
    testing_entries=[]
)

def error_handler(error: Exception, context: Context, next_step: NextStep):
    # TODO Error Handler
    print(error)
    return

# TODO get path from global
pipeline = Pipeline[Context](
    DataPreparationLFW('lfw', 'matchpairsDevTest.csv', 'mismatchpairsDevTest.csv', 5),
    DataPreparation3D('detections', ctx.misc_dir_path + '/shape_predictor_68_face_landmarks.dat'),
    Deep3DCoefficientGenerator()
)

pipeline(ctx, error_handler)