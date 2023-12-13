import shutil
import os
from enum import Enum
from random import choice
from termcolor import cprint

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import OpenTestingEntry


class Expressions(Enum):
    HAPPY = 'happy'
    NORMAL = 'normal'
    SAD = 'sad'
    SLEEPY = 'sleepy'
    SURPRISED = 'surprised'
    WINK = 'wink'


class Lightings(Enum):
    CENTERLIGHT = 'centerlight'
    LEFTLIGHT = 'leftlight'
    RIGHTLIGHT = 'rightlight'


class DataPreparationYale:
    """Process the Yale Dateset and move the wanted files to working dir.
    Also create the open testing entries"""
    def __init__(self, yale_dataset_dir: str,
                 entry_count_limit: int) -> None:
        self._yale_dataset_dir = yale_dataset_dir
        self._entry_count_limit = entry_count_limit


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DataPreparationYale: started', 'cyan')

        entry_counter = 0
        input_faces_count = 0

        for index in range(1, self._entry_count_limit+1):
            # Copy frontal image
            self.copy_image_to_working_dir(context, index, Expressions.NORMAL.value, str(entry_counter) + '_g.png')
            # Loop trough other scenarios and add test entry with (same) frontal image as gallery image
            for expression in Expressions:
                if expression == Expressions.NORMAL:
                    continue
                self.copy_image_to_working_dir(context, index, expression.value, str(input_faces_count) + '_i.png')
                self.add_open_testing_entry(context, input_faces_count, entry_counter, input_faces_count, True, expression=expression.value)
                input_faces_count += 1

            for lighting in Lightings:
                self.copy_image_to_working_dir(context, index, lighting.value, str(input_faces_count) + '_i.png')
                self.add_open_testing_entry(context, input_faces_count, entry_counter, input_faces_count, True, lighting=lighting.value)
                input_faces_count += 1

            entry_counter += 1
            if entry_counter >= self._entry_count_limit+1:
                break

        # Create mis-matches
        mismatch_counter = 0
        for index in range(1, self._entry_count_limit+1):
            # No Files are copied here, since all required image have been moved in the previous for loop
            for expression in Expressions:
                if expression == Expressions.NORMAL:
                    continue
                # Get Random mismatch
                mismatch_entry_key = self.get_random_mismatch_entry_key(context, mismatch_counter, expression.value)
                self.add_open_testing_entry(context, input_faces_count, mismatch_counter, mismatch_entry_key, False, expression=expression.value)
                input_faces_count += 1

            for lighting in Lightings:
                # Get Random mismatch
                mismatch_entry_key = self.get_random_mismatch_entry_key(context, mismatch_counter, lighting.value)
                self.add_open_testing_entry(context, input_faces_count, mismatch_counter, mismatch_entry_key, False, lighting=lighting.value)
                input_faces_count += 1

            mismatch_counter += 1
            if mismatch_counter >= self._entry_count_limit+1:
                break

        print('successfully moved ' + str(entry_counter) + ' images to working dir and prepared testing entry')
        cprint('DataPreparationYale: done', 'green')

        next_step(context)

    def copy_image_to_working_dir(self, context, index: int, scenario: str, new_image_name: str):
        yale_image_path = context.input_dir_path + '/' + self.get_yale_image_path(index, scenario)
        shutil.copy(yale_image_path, context.working_dir_path + '/' + new_image_name)


    # Format of the image file names in the yale dataset:
    # subject01.normal
    #        |    |
    #     index  scenario
    # note: no file ending but all images are gif format
    def get_yale_image_path(self, index: int, scenario: str):
        padded_number = '{:02d}'.format(index)
        return self._yale_dataset_dir + '/subject' + padded_number + '.' + scenario + '.png'


    def add_open_testing_entry(self,
                               context,
                               open_testing_entry_id,
                               gallery_image_id,
                               input_image_id,
                               is_actual_match,
                               expression: str = '',
                               lighting: str = '',):
        context.open_testing_entries[open_testing_entry_id] = OpenTestingEntry(
            gallery_image_file_name=str(gallery_image_id) + '_g.png',
            input_image_file_name=str(input_image_id) + '_i.png',
            is_actual_match=(1 if is_actual_match else 0),
            rotation_angle=0,
            expression=expression,
            lighting=lighting
        )


    def get_random_mismatch_entry_key(self, context, excluded_gallery_image_id: int, scenario: str):
        # todo optimize this filter performance
        filtered_testing_entries = {
            k: v for k, v in context.open_testing_entries.items()
            if v.is_actual_match == 1 and (
                    v.expression == scenario
                    or v.lighting == scenario
            )
        }
        random_item_key = choice(list(filtered_testing_entries))
        if context.open_testing_entries[random_item_key].gallery_image_file_name == str(excluded_gallery_image_id) + '_g.gif':
            return self.get_random_mismatch_entry_key(context, excluded_gallery_image_id, scenario)
        else:
            return random_item_key
