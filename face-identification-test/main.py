import sys
sys.path.append('../deep-3d-face-recon')

from dataclasses import dataclass

from pipeline.pipeline import *
from pipeline_modules.context import Context
from pipeline_modules.data_preparation_3d import DatPreparation3D
from pipeline_modules.deep_3d_coefficient_generator import Deep3DCoefficientGenerator

# Set up Context

ctx = Context(
    input_dir_path='data/input',
    output_dir_path='data/output',
    working_dir_path='data/working',
    misc_dir_path='data/misc',
    deep_3d_coeffs=None
)

def error_handler(error: Exception, context: Context, next_step: NextStep):
    # TODO Error Handler
    return

pipeline = Pipeline[Context](
    DatPreparation3D('detections', ctx.misc_dir_path + '/shape_predictor_68_face_landmarks.dat'),
    Deep3DCoefficientGenerator()
)

pipeline(ctx, error_handler)