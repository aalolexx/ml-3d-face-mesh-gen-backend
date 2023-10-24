from termcolor import cprint
import pandas as pd

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.data_util import get_panda_df_from_context


class ResultTablePKLSaver:
    """Saves the testing result to a pickle file"""

    def __init__(self, pkl_file_name: str) -> None:
        self._pkl_file_name = pkl_file_name


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('ResultTablePKLSaver: started', 'cyan')

        file_path = context.output_dir_path + '/' + self._pkl_file_name
        df = get_panda_df_from_context(context)
        df.to_pickle(file_path)
        print('saved result data to ' + file_path)

        cprint('ResultTablePKLSaver: done', 'green')
        next_step(context)