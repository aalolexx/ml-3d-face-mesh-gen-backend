import shutil
import csv
from termcolor import cprint

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import OpenTestingEntry


class DemoResultPrinter:
    """Prints Results from testing entries table for demo purposes"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DemoResultPrinter: started', 'cyan')

        print("Results from comparison method:")
        for re in context.testing_result_entries:
            print(str(re.method) + ":  \t" + str(round(re.prediction, 4)))

        cprint('DemoResultPrinter: done', 'green')

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