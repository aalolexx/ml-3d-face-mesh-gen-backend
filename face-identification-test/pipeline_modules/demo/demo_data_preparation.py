import shutil
import csv
from termcolor import cprint

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import OpenTestingEntry


class DemoDataPreparation:
    """Data Preparation for Demo Pipeline (only 2 faces will be compared)."""
    def __init__(self, image_a_path: str, image_b_path: str) -> None:
        self._image_a_path = image_a_path
        self._image_b_path = image_b_path


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DemoDataPreparation: started', 'cyan')

        self.copy_image_to_working_dir(context, self._image_a_path, '0_g.jpg')
        self.copy_image_to_working_dir(context, self._image_b_path, '0_i.jpg')
        self.add_open_testing_entry(context, 0)

        print('successfully moved 2 images to working dir and prepared testing entry')
        cprint('DemoDataPreparation: done', 'green')

        next_step(context)


    def copy_image_to_working_dir(self, context, image_path: str, new_image_name):
        input_path = context.input_dir_path + '/' + image_path
        shutil.copy(input_path, context.working_dir_path + '/' + new_image_name)


    def add_open_testing_entry(self, context, id):
        context.open_testing_entries[id] = OpenTestingEntry(
            gallery_image_file_name=str(id) + '_g.jpg',
            input_image_file_name=str(id) + '_i.jpg',
            is_actual_match=0,
            rotation_angle=0,
            expression='',
            lighting=''
        )