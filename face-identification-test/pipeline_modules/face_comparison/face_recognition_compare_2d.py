from termcolor import cprint
import face_recognition

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import TestingResultEntry, FailedTestingEntry
from pipeline_util.enums import ComparisonMethods

class FaceRecognitionCompare2D:
    """Compares two 2D Faces with face_recon's MMOD Method"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('FaceRecognitionCompare2D: started', 'cyan')

        # Loop all open testing entries, get 2d encodings and save comparison result to testing results
        print('comparing ' + str(len(context.open_testing_entries.items())) + ' image pairs')
        for id, testing_entry in context.open_testing_entries.items():
            prediction = 0
            try:
                gallery_image_encoding = context.face_recognition_2d_encodings[
                    testing_entry.gallery_image_file_name.split('.')[0]
                ]
                input_image_encoding = context.face_recognition_2d_encodings[
                    testing_entry.input_image_file_name.split('.')[0]
                ]
                if gallery_image_encoding is not None and input_image_encoding is not None:
                    faces_distance = face_recognition.face_distance([gallery_image_encoding], input_image_encoding)[0]
                    prediction = 1 - faces_distance  # remap the value to have a unified 0 - 1 prediction
                else:
                    cprint('could not 2D face_recognition compare testing entry with id ' + str(id), 'red')
                    context.failed_testing_entries.append(FailedTestingEntry(
                        ComparisonMethods.FACE_RECOGNITION_DISTANCE_2D.name,
                        id,
                        'detection'
                    ))
            except Exception as error:
                cprint('Failed getting the 2D Face encodings on ' + str(id), 'red')
                context.failed_testing_entries.append(FailedTestingEntry(
                    ComparisonMethods.FACE_RECOGNITION_DISTANCE_2D.name,
                    id,
                    'error'
                ))

            context.testing_result_entries.append(TestingResultEntry(
                open_testing_entry_id=id,
                method=ComparisonMethods.FACE_RECOGNITION_DISTANCE_2D.name,
                prediction=prediction
            ))

        cprint('FaceRecognitionCompare2D: done', 'green')
        next_step(context)
