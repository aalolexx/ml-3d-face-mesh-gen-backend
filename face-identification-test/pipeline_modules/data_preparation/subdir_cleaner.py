from termcolor import cprint
import shutil
import os
from pipeline_modules.context import Context
from pipeline.pipeline import NextStep


class SubdirCleaner:
    """Cleans given subddir of Pipeline"""
    def __init__(self, working: bool = False, output: bool = False, custom_path: str = '') -> None:
        self._working = working
        self._output = output
        self._custom_path = custom_path


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('SubdirCleaner: started', 'cyan')

        base_path = None
        if self._working:
            base_path = context.working_dir_path
        if self._output:
            base_path = context.output_dir_path

        if base_path:
            full_path = base_path + '/' + self._custom_path
            shutil.rmtree(full_path)
            os.makedirs(full_path)
            print('cleaned ' + full_path)
        else:
            cprint('Define Working or Output Dir to delete!', 'red')


        cprint('SubdirCleaner: done', 'green')
        next_step(context)
