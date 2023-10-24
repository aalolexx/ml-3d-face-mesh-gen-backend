import sys
sys.path.append('../deep-3d-face-recon')

from pipeline.pipeline import *
from pipeline_modules.context import Context
from pipeline_modules.data_preparation.data_preparation_3d import DataPreparation3D
from pipeline_modules.data_preparation.data_preparation_lfw import DataPreparationLFW
from pipeline_modules.image_analysis.deep_3d_coefficient_generator import Deep3DCoefficientGenerator
from pipeline_modules.image_analysis.face_recon_2d_encoder import FaceRecon2DEncoder
from pipeline_modules.face_comparison.coefficient_based_compare_3d import CoefficientBasedCompare3D
from pipeline_modules.face_comparison.face_recognition_compare_2d import FaceRecognitionCompare2D
from pipeline_modules.result_analysis.roc_curve_plotter import RocCurvePlotter
from pipeline_modules.result_analysis.confusion_matrix_plotter import ConfusionMatrixPlotter
from pipeline_modules.result_analysis.result_table_pkl_reader import ResultTablePKLReader
from pipeline_modules.result_analysis.result_table_pkl_saver import ResultTablePKLSaver
from pipeline_modules.result_analysis.decision_maker import DecisionMaker

# Set up Context

def get_new_context():
    return Context(
        input_dir_path='data/input',
        output_dir_path='data/output',
        working_dir_path='data/working',
        misc_dir_path='data/misc',
        deep_3d_coeffs={},
        face_recognition_2d_encodings={},
        open_testing_entry={},
        testing_result_entries=[],
        panda_dataframe=None
    )

ctx_part_analyzer = get_new_context()


def error_handler(error: Exception, context: Context, next_step: NextStep):
    # TODO Error Handler
    print(error)
    return


# TODO get path from global
pipeline_part_analysis = Pipeline[Context](
    # Data Preparations
    # TODO for dataset data preparation -> define a switch before that and
    DataPreparationLFW('lfw', 'matchpairsDevTest.csv', 'mismatchpairsDevTest.csv', 100),
    DataPreparation3D('detections', ctx_part_analyzer.misc_dir_path + '/shape_predictor_68_face_landmarks.dat'),

    # 3D Analysis
    Deep3DCoefficientGenerator(),

    # 2D Analysis
    FaceRecon2DEncoder(),

    # Face Comparison Methods
    CoefficientBasedCompare3D(),
    # TODO 3D- Viewport Normalization Based
    # TODO 3D- 3D Shape and Texture
    # TODO 2D- MMOD Comparison <-- PRIO 1
    FaceRecognitionCompare2D(),  # TODO - note: MMOD and Hog is for detection only.
    # TODO 2D- HoG Comparison

    # Save Results
    ResultTablePKLSaver('comparison_results.pkl')
)


ctx_part_visualization = get_new_context()
pipeline_part_visualization = Pipeline[Context](
    # Read Results from previous pipeline part
    ResultTablePKLReader('comparison_results.pkl'),

    # Make Decisions from the given predictions
    DecisionMaker(),

    # Result Plotting
    RocCurvePlotter(),
    ConfusionMatrixPlotter()
)


#pipeline_part_analysis(ctx_part_analyzer, error_handler)
pipeline_part_visualization(ctx_part_visualization, error_handler)