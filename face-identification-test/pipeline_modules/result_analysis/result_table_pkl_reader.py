from termcolor import cprint
import pandas as pd

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.data_util import get_panda_df_from_context


class ResultTablePKLReader:
    """Reads the testing result from a pickle file"""

    def __init__(self, pkl_file_name: str) -> None:
        self._pkl_file_name = pkl_file_name


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('ResultTablePKLReader: started', 'cyan')

        df = pd.read_pickle(context.output_dir_path + '/' + self._pkl_file_name)
        context.panda_dataframe = df

        cprint('ResultTablePKLReader: done', 'green')
        next_step(context)