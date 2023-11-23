from termcolor import cprint
from deepface import DeepFace

from pipeline_modules.context import Context, FailedTestingEntry
from pipeline.pipeline import NextStep
from pipeline_modules.context import TestingResultEntry
from pipeline_util.enums import ComparisonMethods

class DeepFaceCompare2D:
    """Compares two 2D Faces with deep face"""
    def __init__(self, model_name: str) -> None:
        if model_name != 'VGG-Face' and model_name != 'Facenet':
            raise Exception('Only VGG-Face and Facenet Models are allowed for the deepface test')
        self._model_name = model_name

    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DeepFaceCompare2D: started', 'cyan')

        method_name = ComparisonMethods.DEEPFACE_DISTANCE_2D_VGG.name
        if self._model_name == 'Facenet':
            method_name = ComparisonMethods.DEEPFACE_DISTANCE_2D_FACENET.name

        # Loop all open testing entries, get actual file images and save comparison result to testing results
        print('comparing ' + str(len(context.open_testing_entry.items())) + ' image pairs')
        for id, testing_entry in context.open_testing_entry.items():
            prediction = 0
            try:
                gallery_image_path = context.working_dir_path + '/' + testing_entry.gallery_image_file_name
                input_image_path = context.working_dir_path + '/' + testing_entry.input_image_file_name
                result = DeepFace.verify(img1_path=gallery_image_path,
                                         img2_path=input_image_path,
                                         detector_backend='mtcnn',
                                         model_name=self._model_name,
                                         enforce_detection=True)
                prediction = 1 - result['distance']
            except Exception as error:
                cprint('Failed getting the 2D Face encodings on ' + str(id), 'red')
                context.failed_testing_entries.append(FailedTestingEntry(
                    method_name,
                    id,
                    'detection'
                ))

            context.testing_result_entries.append(TestingResultEntry(
                open_testing_entry_id=id,
                method=method_name,
                prediction=prediction
            ))

        cprint('DeepFaceCompare2D: done', 'green')
        next_step(context)
