import pandas
from termcolor import cprint
import pandas as pd
from sklearn.metrics import *
import os

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.data_util import *
from pipeline_util.enums import ComparisonMethods


class ScikitReporter:
    """Creates a SciKit Report Summary and exports it to a csv"""
    def __init__(self, export_subdir: str, dataset_name: str, additional_data_query: str = None) -> None:
        self._export_subdir = export_subdir
        self._dataset_name = dataset_name
        self._additional_data_query = additional_data_query


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('ScikitReporter: started', 'cyan')

        pf = context.panda_testing_entries
        if self._additional_data_query:
            pf = pf.query(self._additional_data_query)

        save_path = context.output_dir_path + '/' + self._export_subdir + '/'

        for method in ComparisonMethods:
            pf_cm = pf[(pf.method == method.name)]
            if pf_cm.shape[0] <= 0:
                continue

            y_true = pf_cm['is_actual_match']
            y_pred = pf_cm['decision']
            report = pd.DataFrame({
                'dataset': [self._dataset_name],
                'data query': [self._additional_data_query],
                'method': [method.title],
                'num_pos': [len(y_true == 1)],
                'num_neg': [len(y_true == 0)],
                'accuracy': [accuracy_score(y_true, y_pred)],
                'precision': [precision_score(y_true, y_pred)],
                'recall': [recall_score(y_true, y_pred)],
                'roc_auc': [roc_auc_score(y_true, pf_cm['prediction'])]
            })
            file_path = save_path + self._dataset_name + '_report.csv'
            print_header = not os.path.exists(file_path)
            report.to_csv(file_path, mode='a', index=False, header=print_header)

        cprint('ScikitReporter: done', 'green')
        next_step(context)