from termcolor import cprint
import pandas as pd

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep


class ResultTablePKLReader:
    """Reads the testing result from a pickle file"""

    def __init__(self, pkl_file_name: str, additional_data_query: str = None) -> None:
        self._pkl_file_name = pkl_file_name
        self._additional_data_query = additional_data_query


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('ResultTablePKLReader: started', 'cyan')

        # Testing Result Entries
        df_te = pd.read_pickle(context.output_dir_path + '/' + self._pkl_file_name)
        if self._additional_data_query:
            df_te = df_te.query(self._additional_data_query)
        context.panda_testing_entries = df_te

        # Failed Testing Entries
        df_failed_te = pd.read_pickle(context.output_dir_path + '/failed_' + self._pkl_file_name)
        # Only get failed entries of the *filtered* testing results
        filtered_entry_ids = df_te['open_testing_entry_id'].tolist()
        if len(df_failed_te) > 0:
            df_failed_te = df_failed_te[df_failed_te.open_testing_entry_id.isin(filtered_entry_ids)]

        context.panda_failed_entries = df_failed_te

        cprint('ResultTablePKLReader: done', 'green')
        next_step(context)