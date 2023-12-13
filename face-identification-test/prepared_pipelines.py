import sys
sys.path.append('../deep-3d-face-recon')

from pipeline.pipeline import *
from pipeline_modules.context import Context
from pipeline_modules.data_preparation.subdir_cleaner import SubdirCleaner
from pipeline_modules.data_preparation.data_preparation_3d import DataPreparation3D
from pipeline_modules.data_preparation.data_preparation_lfw import DataPreparationLFW
from pipeline_modules.data_preparation.data_preparation_yale import DataPreparationYale
from pipeline_modules.data_preparation.data_preparation_yale import Expressions as YaleExpressions
from pipeline_modules.data_preparation.data_preparation_pie import DataPreparationPIE
from pipeline_modules.data_preparation.data_preparation_pie import Expressions as PIEExpressions
from pipeline_modules.data_preparation.data_preparation_pie import Lightings as PIELightings
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
from pipeline_modules.result_analysis.scikit_reporter import ScikitReporter
from pipeline_util.enums import ComparisonFramework, Datasets


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
        open_testing_entries={},
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
        SubdirCleaner(working=True),
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
    yale_query = 'expression == "' + YaleExpressions.HAPPY.value + '"'
    pie_query = 'rotation_angle == 0'

    return Pipeline[Context](
        SubdirCleaner(output=True, custom_path='t1'),
        # First Use Yale Test Data here
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + Datasets.YALE.name + '.pkl',
            additional_data_query=yale_query),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(export_subdir='t1', dataset_name=Datasets.YALE.name),
        ConfusionMatrixPlotter(export_subdir='t1', dataset_name=Datasets.YALE.name),

        # Then the same for PIE
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + Datasets.MULTIPIE.name + '.pkl',
            additional_data_query=pie_query),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(export_subdir='t1', dataset_name=Datasets.MULTIPIE.name),
        ConfusionMatrixPlotter(export_subdir='t1', dataset_name=Datasets.MULTIPIE.name),
    )


def get_pipeline_t2():
    t2_query = 'expression =="' + PIEExpressions.NORMAL.value + '"' \
               'and lighting == "' + PIELightings.CENTERLIGHT.value + '"'
    query_extreme_rotations = 'rotation_angle <= -45 or rotation_angle > 0'
    query_normal_rotations = 'rotation_angle > -45 or rotation_angle < 45'

    return Pipeline[Context](
        SubdirCleaner(output=True, custom_path='t2'),
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + Datasets.MULTIPIE.name + '.pkl',
            additional_data_query=t2_query),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        ScikitReporter(export_subdir='t2',
                       dataset_name=Datasets.MULTIPIE.name),
        ScikitReporter(export_subdir='t2',
                       dataset_name=Datasets.MULTIPIE.name,
                       additional_data_query=query_extreme_rotations),
        ScikitReporter(export_subdir='t2',
                       dataset_name=Datasets.MULTIPIE.name,
                       additional_data_query=query_normal_rotations),
        RocCurvePlotter(export_subdir='t2',
                        dataset_name=Datasets.MULTIPIE.name),
        ConfusionMatrixPlotter(export_subdir='t2',
                               dataset_name=Datasets.MULTIPIE.name),
        CategorizedBasedBarPlotter(group_by_category='rotation_angle',
                                   export_subdir='t2',
                                   dataset_name=Datasets.MULTIPIE.name),
        FailedEntriesBarPlotter(export_subdir='t2', dataset_name=Datasets.MULTIPIE.name)
    )


def get_pipeline_t3():
    return Pipeline[Context](
        SubdirCleaner(output=True, custom_path='t3'),
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + Datasets.YALE.name + '.pkl'),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(export_subdir='t3', dataset_name=Datasets.YALE.name),
        CategorizedBasedBarPlotter(group_by_category='lighting',
                                   export_subdir='t3',
                                   dataset_name=Datasets.YALE.name),
        FailedEntriesBarPlotter(export_subdir='t3', dataset_name=Datasets.YALE.name)
    )


def get_pipeline_t4():
    return Pipeline[Context](
        SubdirCleaner(output=True, custom_path='t4'),
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + Datasets.YALE.name + '.pkl'),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(export_subdir='t4', dataset_name=Datasets.YALE.name),
        CategorizedBasedBarPlotter(group_by_category='expression',
                                   export_subdir='t4',
                                   dataset_name=Datasets.YALE.name),
        FailedEntriesBarPlotter(export_subdir='t4', dataset_name=Datasets.YALE.name)
    )


def get_pipeline_t5():
    t5_query = 'expression =="' + PIEExpressions.NORMAL.value + '"' \
               'and lighting != "' + PIELightings.CENTERLIGHT.value + '"'
    return Pipeline[Context](
        SubdirCleaner(output=True, custom_path='t5'),
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + Datasets.MULTIPIE.name + '.pkl',
            additional_data_query=t5_query),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(export_subdir='t5', dataset_name=Datasets.MULTIPIE.name),
        CategorizedBasedBarPlotter(group_by_category='rotation_angle',
                                   export_subdir='t5',
                                   dataset_name=Datasets.MULTIPIE.name),
        CategorizedBasedBarPlotter(group_by_category='lighting',
                                   export_subdir='t5',
                                   dataset_name=Datasets.MULTIPIE.name),
        FailedEntriesBarPlotter(export_subdir='t5', dataset_name=Datasets.MULTIPIE.name)
    )


def get_pipeline_t6():
    t6_query = 'expression !="' + PIEExpressions.NORMAL.value + '"' \
               'and lighting != "' + PIELightings.CENTERLIGHT.value + '"'
    return Pipeline[Context](
        SubdirCleaner(output=True, custom_path='t6'),
        ResultTablePKLReader(
            pkl_file_name='comparison_results_' + Datasets.MULTIPIE.name + '.pkl',
            additional_data_query=t6_query),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(export_subdir='t6', dataset_name=Datasets.MULTIPIE.name),
        CategorizedBasedBarPlotter(group_by_category='rotation_angle',
                                   export_subdir='t6',
                                   dataset_name=Datasets.MULTIPIE.name),
        CategorizedBasedBarPlotter(group_by_category='lighting',
                                   export_subdir='t6',
                                   dataset_name=Datasets.MULTIPIE.name),
        CategorizedBasedBarPlotter(group_by_category='expression',
                                   export_subdir='t6',
                                   dataset_name=Datasets.MULTIPIE.name),
        FailedEntriesBarPlotter(export_subdir='t6', dataset_name=Datasets.MULTIPIE.name)
    )


def get_pipeline_t7():
    return Pipeline[Context](
        SubdirCleaner(output=True, custom_path='t7'),
        # Read Results from previous pipeline part
        ResultTablePKLReader('comparison_results_' + Datasets.LFW.name + '.pkl'),
        # Make Decisions from the given predictions
        DecisionMaker(),
        # Result Plotting
        RocCurvePlotter(export_subdir='t7', dataset_name=Datasets.LFW.name),
        ConfusionMatrixPlotter(export_subdir='t7', dataset_name=Datasets.LFW.name),
        FailedEntriesBarPlotter(export_subdir='t7', dataset_name=Datasets.LFW.name)
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
    ctx = get_new_context()
    pipeline = get_pipeline_t1()
    pipeline(ctx, error_handler)


def run_t2():
    ctx = get_new_context()
    pipeline_pie = get_pipeline_t2()
    pipeline_pie(ctx, error_handler)


def run_t3():
    ctx = get_new_context()
    pipeline_yale = get_pipeline_t3()
    pipeline_yale(ctx, error_handler)


def run_t4():
    ctx = get_new_context()
    pipeline_yale = get_pipeline_t4()
    pipeline_yale(ctx, error_handler)


def run_t5():
    ctx = get_new_context()
    pipeline_pie = get_pipeline_t5()
    pipeline_pie(ctx, error_handler)


def run_t6():
    ctx = get_new_context()
    pipeline_pie = get_pipeline_t6()
    pipeline_pie(ctx, error_handler)


def run_t7():
    ctx = get_new_context()
    pipeline_lfw = get_pipeline_t7()
    pipeline_lfw(ctx, error_handler)
