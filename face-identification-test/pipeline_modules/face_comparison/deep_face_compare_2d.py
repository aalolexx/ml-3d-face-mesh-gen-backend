from termcolor import cprint
from deepface import DeepFace

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import TestingResultEntry
from pipeline_util.enums import ComparisonMethods

class DeepFaceCompare2D:
    """Compares two 2D Faces with deep face"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DeepFaceCompare2D: started', 'cyan')

        # Loop all open testing entries, get actual file images and save comparison result to testing results
        print('comparing ' + str(len(context.open_testing_entry.items())) + ' image pairs')
        for id, testing_entry in context.open_testing_entry.items():
            faces_distance = 0
            try:
                gallery_image_path = context.working_dir_path + '/' + testing_entry.gallery_image_file_name
                input_image_path = context.working_dir_path + '/' + testing_entry.input_image_file_name
                result = DeepFace.verify(img1_path=gallery_image_path,
                                         img2_path=input_image_path,
                                         detector_backend='mtcnn')
                faces_distance = 1 - result['distance']
            except Exception as error:
                cprint('Failed getting the 2D Face encodings on ' + str(id), 'red')

            context.testing_result_entries.append(TestingResultEntry(
                open_testing_entry_id=id,
                method=ComparisonMethods.DEEPFACE_DISTANCE_2D.name,
                prediction=faces_distance
            ))

        cprint('DeepFaceCompare2D: done', 'green')
        next_step(context)