from termcolor import cprint
import face_recognition
from deepface import DeepFace
from enum import Enum

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import TestingResultEntry
from pipeline_util.enums import ComparisonMethods, ComparisonFramework


class VPNImageCompare:
    """Compares the vpn generated images. bidirectional or unidirectional, depending on param setting"""
    def __init__(self, comparison_framework: ComparisonFramework,
                 bidirectional: bool = True) -> None:
        self._bidirectional = bidirectional
        self._comparison_framework = comparison_framework


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('VPNImageCompare: started', 'cyan')

        prediction = 0
        required_method = ComparisonMethods.UNIDIRECTIONAL_VPN_COMPARE.name
        if self._bidirectional:
            required_method = ComparisonMethods.BIDIRECTIONAL_VPN_COMPARE.name

        # Loop all open testing entries, get 2d encodings and save comparison result to testing results
        print('comparing ' + str(len(context.open_testing_entry.items())) + ' image pairs')
        for id, testing_entry in context.open_testing_entry.items():
            try:
                if self._comparison_framework == ComparisonFramework.FACE_RECOGNITION:
                    prediction = self.face_recognition_compare(context, testing_entry)
                else:
                    prediction = self.deepface_compare(context, testing_entry)
            except Exception as error:
                cprint('Failed getting the 2D Face encodings on ' + str(id), 'red')

            context.testing_result_entries.append(TestingResultEntry(
                open_testing_entry_id=id,
                method=required_method,
                prediction=prediction
            ))

        cprint('VPNImageCompare: done', 'green')
        next_step(context)

    def face_recognition_compare(self, context, testing_entry):
        prediction = 0

        input_vpn_image_encoding = context.face_recognition_2d_encodings[
            'vpn_' + testing_entry.input_image_file_name.split('.')[0]
        ]

        # Get either "original" image or vpn image, depending on VPN comparison method
        if self._bidirectional:
            required_gallery_image_encoding = context.face_recognition_2d_encodings[
                'vpn_' + testing_entry.gallery_image_file_name.split('.')[0]
            ]
        else:
            required_gallery_image_encoding = context.face_recognition_2d_encodings[
                testing_entry.gallery_image_file_name.split('.')[0]
            ]

        if required_gallery_image_encoding is not None and input_vpn_image_encoding is not None:
            faces_distance = face_recognition.face_distance([required_gallery_image_encoding], input_vpn_image_encoding)[0]
            prediction = 1 - faces_distance
        else:
            cprint('could not 2D VPN (face_recognition) compare testing entry with '
                   + testing_entry.gallery_image_file_name + '/' + testing_entry.input_image_file_name, 'red')

        return prediction


    # In Tests this turned out to be less accurate on VPN Images, refer to Bachelorthesis to see the tests
    def deepface_compare(self, context, testing_entry):
        if self._bidirectional:
            gallery_image_path = context.working_dir_path + '/vpn_' + testing_entry.gallery_image_file_name
        else:
            gallery_image_path = context.working_dir_path + '/' + testing_entry.gallery_image_file_name

        input_image_path = context.working_dir_path + '/vpn_' + testing_entry.input_image_file_name
        result = DeepFace.verify(img1_path=gallery_image_path,
                                 img2_path=input_image_path,
                                 detector_backend='mtcnn')
        return 1 - result['distance']
