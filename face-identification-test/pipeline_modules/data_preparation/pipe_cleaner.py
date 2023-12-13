from termcolor import cprint
import shutil
import os
from pipeline_modules.context import Context
from pipeline.pipeline import NextStep


class PipeCleaner:
    """Cleans Working Dir of Pipeline"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('PipeCleaner: started', 'cyan')

        shutil.rmtree(context.working_dir_path + '/')
        os.makedirs(context.working_dir_path)

        cprint('PipeCleaner: done', 'green')
        next_step(context)
