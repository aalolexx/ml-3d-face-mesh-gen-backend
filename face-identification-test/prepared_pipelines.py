import sys
from pipeline.pipeline import *
from pipeline_modules.context import Context
from pipeline_modules.data_preparation.data_preparation_3d import DataPreparation3D
from pipeline_modules.data_preparation.data_preparation_lfw import DataPreparationLFW
from pipeline_modules.data_preparation.data_preparation_yale import DataPreparationYale, Scenarios
from pipeline_modules.data_preparation.data_preparation_pie import DataPreparationPIE
from pipeline_modules.image_analysis.deep_3d_coefficient_generator import Deep3DCoefficientGenerator
from pipeline_modules.image_analysis.vpn_image_creator import VPNImageCreator
from pipeline_modules.image_analysis.face_recon_2d_encoder import FaceRecon2DEncoder
from pipeline_modules.face_comparison.coefficient_based_compare_3d import CoefficientBasedCompare3D
from pipeline_modules.face_comparison.face_recognition_compare_2d import FaceRecognitionCompare2D
from pipeline_modules.face_comparison.deep_face_compare_2d import DeepFaceCompare2D
from pipeline_modules.face_comparison.vpn_image_compare import VPNImageCompare
from pipeline_modules.result_analysis.roc_curve_plotter import RocCurvePlotter
from pipeline_modules.result_analysis.categorized_based_bar_plotter import CategorizedBasedBarPlotter
from pipeline_modules.result_analysis.confusion_matrix_plotter import ConfusionMatrixPlotter
from pipeline_modules.result_analysis.failed_entries_bar_plotter import FailedEntriesBarPlotter
from pipeline_modules.result_analysis.result_table_pkl_reader import ResultTablePKLReader
from pipeline_modules.result_analysis.result_table_pkl_saver import ResultTablePKLSaver
from pipeline_modules.result_analysis.decision_maker import DecisionMaker
from pipeline_util.enums import ComparisonFramework, Datasets

sys.path.append('../deep-3d-face-recon')


#
# Set up Context and error handler
#


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
        failed_testing_entries=[],
        panda_testing_entries=None,
        panda_failed_entries=None
    )


def error_handler(error: Exception, context: Context, next_step: NextStep):
    # TODO Error Handler
    print(error)
    raise ValueError(error) from error


#
# Prepared Testing Piplines
# Data Testing & Visualization
#


def get_test_pipeline_for_dataset(dataset_prep_module, pkl_suffix):
    return Pipeline[Context](
        # Data Preparations
        dataset_prep_module,
        DataPreparation3D('detections'),

        # 3D Analysis
        Deep3DCoefficientGenerator(),

        # Generate Viewpoint Normalized Images
        VPNImageCreator('avg_person.png'),

        # 2D Analysis
        FaceRecon2DEncoder(include_vpn_images=True),

        # Face Comparison Methods
        CoefficientBasedCompare3D(),
        VPNImageCompare(bidirectional=True, comparison_framework=ComparisonFramework.FACE_RECOGNITION),
        VPNImageCompare(bidirectional=False, comparison_framework=ComparisonFramework.FACE_RECOGNITION),
        FaceRecognitionCompare2D(),
        DeepFaceCompare2D(model_name='VGG-Face'),
        DeepFaceCompare2D(model_name='Facenet'),

        # Save Results
        ResultTablePKLSaver('comparison_results_' + pkl_suffix + '.pkl')
    )


def get_pipeline_t1():
    # TODO
    return None


def get_pipeline_t2(pkl_suffix: str):
    return Pipeline[Context](
        # Read Results from previous pipeline part
        ResultTablePKLReader('comparison_results_' + pkl_suffix + '.pkl'),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(),
        CategorizedBasedBarPlotter('rotation_angle'),
        FailedEntriesBarPlotter()
    )


def get_pipeline_t3(pkl_suffix: str):
    t3_data_query = 'scenario == "' + Scenarios.CENTERLIGHT.value + '" | '\
                  + 'scenario == "' + Scenarios.LEFTLIGHT.value + '" | '\
                  + 'scenario == "' + Scenarios.RIGHTLIGHT.value + '"'

    return Pipeline[Context](
        # Read Results from previous pipeline part
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + pkl_suffix + '.pkl',
            additional_data_query=t3_data_query),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(),
        CategorizedBasedBarPlotter('scenario'),
        FailedEntriesBarPlotter()
    )


def get_pipeline_t4(pkl_suffix: str):
    t4_data_query = 'scenario == "' + Scenarios.NORMAL.value + '" | ' \
                    + 'scenario == "' + Scenarios.HAPPY.value + '" | ' \
                    + 'scenario == "' + Scenarios.SAD.value + '" | ' \
                    + 'scenario == "' + Scenarios.SLEEPY.value + '" | ' \
                    + 'scenario == "' + Scenarios.SURPRISED.value + '" | ' \
                    + 'scenario == "' + Scenarios.WINK.value + '"'

    return Pipeline[Context](
        # Read Results from previous pipeline part
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + pkl_suffix + '.pkl',
            additional_data_query=t4_data_query),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(),
        CategorizedBasedBarPlotter('scenario'),
        FailedEntriesBarPlotter()
    )


def get_pipeline_t5():
    # TODO
    return None


def get_pipeline_t6():
    # TODO
    return None


def get_pipeline_t7(pkl_suffix: str):
    return Pipeline[Context](
        # Read Results from previous pipeline part
        ResultTablePKLReader('comparison_results_' + pkl_suffix + '.pkl'),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(),  # zb pf["rotation_angle"] == -60
        ConfusionMatrixPlotter(''),
        FailedEntriesBarPlotter()
    )


#
# Run the prepared Pipelines
#

def run_analyzer(dataset: Datasets, test_entry_count: int, test_all_available: bool = False):
    data_prep_module = None

    if dataset == Datasets.YALE:
        if test_all_available:
            test_entry_count = Datasets.YALE.value
        data_prep_module = DataPreparationYale('yale-face-database', test_entry_count)

    elif dataset == Datasets.LFW:
        if test_all_available:
            test_entry_count = Datasets.LFW.value
        data_prep_module = DataPreparationLFW('lfw', 'matchpairsDevTest.csv', 'mismatchpairsDevTest.csv', test_entry_count)

    elif dataset == Datasets.MULTIPIE:
        if test_all_available:
            test_entry_count = Datasets.MULTIPIE.value
        data_prep_module = DataPreparationPIE('multi-pie', test_entry_count)

    ctx = get_new_context()
    pipeline = get_test_pipeline_for_dataset(data_prep_module, dataset.name)
    pipeline(ctx, error_handler)


def run_t1():
    # TODO
    return


def run_t2():
    ctx = get_new_context()
    pipeline_pie = get_pipeline_t2(Datasets.MULTIPIE.name)
    pipeline_pie(ctx, error_handler)


def run_t3():
    ctx = get_new_context()
    pipeline_yale = get_pipeline_t3(Datasets.YALE.name)
    pipeline_yale(ctx, error_handler)


def run_t4():
    ctx = get_new_context()
    pipeline_yale = get_pipeline_t4(Datasets.YALE.name)
    pipeline_yale(ctx, error_handler)
    return


def run_t5():
    # TODO
    return


def run_t6():
    # TODO
    return


def run_t7():
    ctx = get_new_context()
    pipeline_lfw = get_pipeline_t7(Datasets.LFW.name)
    pipeline_lfw(ctx, error_handler)