import shutil
import csv
import os
from termcolor import cprint

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_modules.context import TestingEntry


class DataPreparationLFW:
    """Finds Pairs and mis Pairs from LFW Datasets and moves
    them to the working dir with the correct naming"""
    def __init__(self, lfw_dataset_dir: str,
                 csv_matchpairs_path: str,
                 csv_mismatchpairs_path: str,
                 entry_count_limit: int) -> None:
        self._lfw_dataset_dir = lfw_dataset_dir
        self._csv_matchpairs_path = csv_matchpairs_path
        self._csv_mismatchpairs_path = csv_mismatchpairs_path
        self._entry_count_limit = entry_count_limit


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DataPreparationLFW: started', 'cyan')

        # clean working dir
        # TODO Make a cleanup pipeline module
        shutil.rmtree(context.working_dir_path + '/')
        os.makedirs(context.working_dir_path)

        complete_lfw_dataset_dir = context.input_dir_path + '/' + self._lfw_dataset_dir
        entry_counter = 0

        # Loop trough all matches, copy file and prepare testing entry
        with open(complete_lfw_dataset_dir + '/' + self._csv_matchpairs_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip header line
            i = 0
            for row in csv_reader:
                self.copy_image_to_working_dir(context, row[0], int(row[1]), str(entry_counter) + '_g.jpg')
                self.copy_image_to_working_dir(context, row[0], int(row[2]), str(entry_counter) + '_i.jpg')
                context.testing_entries.append(TestingEntry(
                    id=entry_counter,
                    method=None,
                    gallery_image_file_name=str(entry_counter) + '_g.jpg',
                    input_image_file_name=str(entry_counter) + '_i.jpg',
                    is_actual_match=True,
                    prediction=None
                ))
                entry_counter += 1
                i += 1
                if i >= self._entry_count_limit:
                    break

        # Loop trough all *MIS* matches, copy file and prepare testing entry
        with open(complete_lfw_dataset_dir + '/' + self._csv_mismatchpairs_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip header line
            i = 0
            for row in csv_reader:
                self.copy_image_to_working_dir(context, row[0], int(row[1]), str(entry_counter) + '_g.jpg')
                self.copy_image_to_working_dir(context, row[2], int(row[3]), str(entry_counter) + '_i.jpg')
                context.testing_entries.append(TestingEntry(
                    id=entry_counter,
                    method=None,
                    gallery_image_file_name=str(entry_counter) + '_g.jpg',
                    input_image_file_name=str(entry_counter) + '_i.jpg',
                    is_actual_match=False,
                    prediction=None
                ))
                entry_counter += 1
                i += 1
                if i >= self._entry_count_limit:
                    break

        print('successfully moved ' + str(entry_counter) + ' images to working dir and prepared testing entry')
        cprint('DataPreparationLFW: done', 'green')

        next_step(context)


    def copy_image_to_working_dir(self, context, name: str, num: int, new_image_name: str):
        lfw_image_path = context.input_dir_path + '/' + self.get_lfw_image_path(name, num)
        shutil.copy(lfw_image_path, context.working_dir_path + '/' + new_image_name)

    def get_lfw_image_path(self, name: str, num: int):
        padded_number = '{:04d}'.format(num)
        return self._lfw_dataset_dir + '/lfw-deepfunneled/lfw-deepfunneled/'\
                + name + '/' + name + '_' + padded_number + '.jpg'