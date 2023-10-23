from termcolor import cprint
import face_recognition

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import TestingResultEntry
from pipeline_util.enums import ComparisonMethods

class FaceRecognitionCompare2D:
    """Compares two 2D Faces with face_recon's MMOD Method"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('FaceRecognitionCompare2D: started', 'cyan')

        # Loop all open testing entries, get 2d encodings and save comparison result to testing results
        print('comparing ' + str(len(context.open_testing_entry.items())) + ' image pairs')
        for id, testing_entry in context.open_testing_entry.items():
            gallery_image_encoding = context.face_recognition_2d_encodings[str(id) + '_g']
            input_image_encoding = context.face_recognition_2d_encodings[str(id) + '_i']
            faces_distance = face_recognition.face_distance([gallery_image_encoding], input_image_encoding)
            context.testing_result_entries.append(TestingResultEntry(
                open_testing_entry_id=id,
                method=ComparisonMethods.FACE_RECOGNITION_DISTANCE_2D,
                prediction=faces_distance
            ))

        cprint('FaceRecognitionCompare2D: done', 'green')
        next_step(context)
