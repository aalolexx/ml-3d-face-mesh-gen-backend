import shutil
import os
from random import choice
from termcolor import cprint
from enum import Enum

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import OpenTestingEntry


class Expressions(Enum):
    NORMAL = '01'
    HAPPY = '02'


class Lightings(Enum):
    CENTERLIGHT = '06'
    LEFTLIGHT = '13'
    RIGHTLIGHT = '01'
    DARK = '00'


class DataPreparationPIE:
    """Process the PIE Dateset and move the wanted files to working dir.
    Also prepares openTestingEntries List."""
    def __init__(self, pie_dataset_dir: str,
                 entry_count_limit: int) -> None:
        self._pie_dataset_dir = pie_dataset_dir
        self._entry_count_limit = entry_count_limit
        self._frontal_camera = '051'
        self._side_cameras = {
            '010': 75,
            '200': 60,
            '190': 45,
            '041': 30,
            '050': 15,
            '051': 0,
            '140': -15,
            '130': -30,
            '080': -45,
            '090': -60,
            '120': -75
        }


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DataPreparationPIE: started', 'cyan')

        complete_pie_dataset_dir = context.input_dir_path + '/' + self._pie_dataset_dir
        entry_counter = 0
        input_faces_count = 0

        # Loop through all faces, add multiple Angles Input Images to a Gallery Input and add testing entry (as match)
        faces_folder_names = os.listdir(complete_pie_dataset_dir)
        for face in faces_folder_names:
            # Copy frontal image

            self.copy_image_to_working_dir(context,
                                           face,
                                           self._frontal_camera,
                                           Expressions.NORMAL.value,
                                           Lightings.CENTERLIGHT.value,
                                           str(entry_counter) + '_g.png')
            # Loop trough other angles and add test entry with (same) frontal image as gallery image
            for camera, angle in self._side_cameras.items():
                for expression in Expressions:
                    for lighting in Lightings:
                        self.copy_image_to_working_dir(context,
                                                       face,
                                                       camera,
                                                       expression.value,
                                                       lighting.value,
                                                       str(input_faces_count) + '_i.png')
                        self.add_open_testing_entry(context,
                                                    input_faces_count,
                                                    entry_counter,
                                                    input_faces_count,
                                                    True,
                                                    angle,
                                                    expression.value,
                                                    lighting.value)
                        input_faces_count += 1

            entry_counter += 1
            if entry_counter >= self._entry_count_limit:
                break

        # Create mis-matches
        mismatch_counter = 0
        for face in faces_folder_names:
            # No Files are copied here, since all required image have been moved in the previous for loop
            for camera, angle in self._side_cameras.items():
                for expression in Expressions:
                    for lighting in Lightings:
                        # Get Random mismatch
                        mismatch_entry_key = self.get_random_mismatch_entry_key(context,
                                                                                mismatch_counter,
                                                                                angle,
                                                                                expression.value,
                                                                                lighting.value)
                        self.add_open_testing_entry(context,
                                                    input_faces_count,
                                                    mismatch_counter,
                                                    mismatch_entry_key,
                                                    False,
                                                    angle,
                                                    expression.value,
                                                    lighting.value)
                        input_faces_count += 1
            mismatch_counter += 1
            if mismatch_counter >= self._entry_count_limit:
                break

        print('successfully moved ' + str(entry_counter+mismatch_counter) + ' faces to working dir and prepared testing entry')
        cprint('DataPreparationPIE: done', 'green')

        next_step(context)


    def copy_image_to_working_dir(self,
                                  context,
                                  face: str,
                                  camera: str,
                                  expression: str,
                                  lighting: str,
                                  new_image_name: str):
        pie_image_path = context.input_dir_path + '/' + self.get_pie_image_path(face, camera, expression, lighting)
        shutil.copy(pie_image_path, context.working_dir_path + '/' + new_image_name)


    # Format of the image file names in the pie dataset:
    # 001_01_01_010_17_crop_128.png
    #  |   |  |  |  |
    # Face |Expr.|  Lighting
    #   Session Camera
    def get_pie_image_path(self, face: str, camera: str, expression: str, lighting: str):
        return self._pie_dataset_dir + '/' + face + '/' \
                + face + '_01_' + expression + '_' + camera + '_' + lighting + '_crop_128.png'


    def add_open_testing_entry(self,
                               context,
                               open_testing_entry_id,
                               gallery_image_id,
                               input_image_id,
                               is_actual_match,
                               rotation_angle,
                               expression,
                               lighting):
        context.open_testing_entries[open_testing_entry_id] = OpenTestingEntry(
            gallery_image_file_name=str(gallery_image_id) + '_g.png',
            input_image_file_name=str(input_image_id) + '_i.png',
            is_actual_match=(1 if is_actual_match else 0),
            rotation_angle=rotation_angle,
            expression=expression,
            lighting=lighting
        )


    def get_random_mismatch_entry_key(self, context, excluded_gallery_image_id: int, angle: int, expression: str, lighting: str):
        # todo optimize this filter performance
        filtered_testing_entries = {
            k: v for k, v in context.open_testing_entries.items()
            if v.is_actual_match == 1 and (
                    v.rotation_angle == angle
                    and v.expression == expression
                    and v.lighting == lighting
            )
        }
        random_item_key = choice(list(filtered_testing_entries))
        if context.open_testing_entries[random_item_key].gallery_image_file_name == str(excluded_gallery_image_id) + '_g.png':
            return self.get_random_mismatch_entry_key(context, excluded_gallery_image_id, angle, expression, lighting)
        else:
            return random_item_key
