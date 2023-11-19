from termcolor import cprint
import face_recognition

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import TestingResultEntry
from pipeline_util.enums import ComparisonMethods

class VPNImageCompare:
    """Compares the vpn generated images. bidirectional or unidirectional, depending on param setting"""
    def __init__(self, bidirectional=True) -> None:
        self._bidirectional = bidirectional


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('VPNImageCompare: started', 'cyan')

        # Loop all open testing entries, get 2d encodings and save comparison result to testing results
        print('comparing ' + str(len(context.open_testing_entry.items())) + ' image pairs')
        for id, testing_entry in context.open_testing_entry.items():
            faces_distance = 0
            try:
                # TODO make util function and use both here and in face_recon 2D compare
                # TODO use bidirectional Param

                gallery_vpn_image_encoding = context.face_recognition_2d_encodings[
                    'vpn_' + testing_entry.gallery_image_file_name.split('.')[0]
                ]
                input_vpn_image_encoding = context.face_recognition_2d_encodings[
                    'vpn_' + testing_entry.input_image_file_name.split('.')[0]
                ]
                if gallery_vpn_image_encoding is not None and input_vpn_image_encoding is not None:
                    faces_distance = face_recognition.face_distance([gallery_vpn_image_encoding], input_vpn_image_encoding)[0]
                    faces_distance = 1 - faces_distance  # remap the value to have a unified 0 - 1 prediction
                else:
                    cprint('could not 2D VPN compare testing entry with id ' + str(id), 'red')
            except Exception as error:
                cprint('Failed getting the 2D Face encodings on ' + str(id), 'red')

            context.testing_result_entries.append(TestingResultEntry(
                open_testing_entry_id=id,
                method=ComparisonMethods.BIDIRECTIONAL_VPN_COMPARE.name,
                prediction=faces_distance
            ))

        cprint('VPNImageCompare: done', 'green')
        next_step(context)
