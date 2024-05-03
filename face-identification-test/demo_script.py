import sys
sys.path.append('../deep-3d-face-recon')

from pipeline_modules.data_preparation.data_preparation_3d import DataPreparation3D
from pipeline_modules.data_preparation.subdir_cleaner import SubdirCleaner
from pipeline_modules.face_comparison.coefficient_based_compare_3d import CoefficientBasedCompare3D
from pipeline_modules.face_comparison.deep_face_compare_2d import DeepFaceCompare2D
from pipeline_modules.face_comparison.face_recognition_compare_2d import FaceRecognitionCompare2D
from pipeline_modules.face_comparison.vpn_image_compare import VPNImageCompare
from pipeline_modules.image_analysis.deep_3d_coefficient_generator import Deep3DCoefficientGenerator
from pipeline_modules.image_analysis.face_recon_2d_encoder import FaceRecon2DEncoder
from pipeline_modules.image_analysis.vpn_image_creator import VPNImageCreator
from pipeline_util.enums import ComparisonFramework
from prepared_pipelines import get_new_context, error_handler
from pipeline.pipeline import *
from pipeline_modules.demo.demo_data_preparation import DemoDataPreparation
from pipeline_modules.demo.demo_result_printer import DemoResultPrinter



# 1. read two images
# 2. Do 3D Preparation
# 3. Do all Comparison Methods
# 4. print results

ctx = get_new_context()

def run_demo(img_a, img_b):
    ctx = get_new_context()
    demo_pipeline = Pipeline[Context](
        # data prep
        SubdirCleaner(working=True),
        DemoDataPreparation(img_a, img_b),
        DataPreparation3D('detections'),
        Deep3DCoefficientGenerator(),
        VPNImageCreator('avg_person.png'),
        FaceRecon2DEncoder(include_vpn_images=True),
        # comparison
        CoefficientBasedCompare3D(),
        VPNImageCompare(bidirectional=True, comparison_framework=ComparisonFramework.FACE_RECOGNITION),
        VPNImageCompare(bidirectional=False, comparison_framework=ComparisonFramework.FACE_RECOGNITION),
        FaceRecognitionCompare2D(),
        DeepFaceCompare2D(model_name='VGG-Face'),
        DeepFaceCompare2D(model_name='Facenet'),
        # Print the results
        DemoResultPrinter()
    )
    demo_pipeline(ctx, error_handler)

# METHODS threshold according to T5
# 3dmm coefficient  0.7120230793952942
# 3dmm vpn bi       0.4875996092282171
# 3dmm vpn uni      0.30174234561004576
# face_recon 2d     0.41771562244219607
# deep facenet      0.29704224104151744
# deep vgg          0.6262551861354412

run_demo('demo/img_c1.png', 'demo/img_c4.png')