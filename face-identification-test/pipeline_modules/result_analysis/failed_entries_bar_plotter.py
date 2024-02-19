from termcolor import cprint
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods
from pipeline_util.plot_util import *


class FailedEntriesBarPlotter:
    """Plots a Bar chart showing the failed entries per method
    from the context.panda_testing_entries table"""
    def __init__(self, export_subdir: str, dataset_name: str) -> None:
        self._export_subdir = export_subdir
        self._dataset_name = dataset_name


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('FailedEntriesBarPlotter: started', 'cyan')

        pf_failed_testing_entries = context.panda_failed_entries

        failed_method_counter = []
        for method in ComparisonMethods:
            fail_count = 0
            if len(pf_failed_testing_entries) > 0:
                fail_count = pf_failed_testing_entries['failed_method'].str.count(method.name).sum()
            failed_method_counter.append({
                'method': method.title,
                'count': fail_count
            })

        pd_data = pd.DataFrame(failed_method_counter)

        save_path = context.output_dir_path + '/' + self._export_subdir + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        set_sns_style(sns)

        method_palette = [m.color for m in ComparisonMethods]

        ax = sns.barplot(x='count',
                    y='method',
                    data=pd_data,
                    palette=method_palette,
                    orient='h')

        for bar in ax.patches:
            xval = bar.get_width()
            plt.text(xval+5, bar.get_y() + bar.get_height()/2, int(xval), ha='left', va='center')

        plt.ylabel('')
        plt.title('Failed Entries (using ' + self._dataset_name + ')')
        save_fig(plt, save_path + self._dataset_name + '_count_failed_entries.png')
        plt.close()

        cprint('FailedEntriesBarPlotter: done', 'green')
        next_step(context)
