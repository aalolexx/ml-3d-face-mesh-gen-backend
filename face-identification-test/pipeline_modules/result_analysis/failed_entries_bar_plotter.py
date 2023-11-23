from termcolor import cprint
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods

class FailedEntriesBarPlotter:
    """Plots a Bar chart showing the failed entrys per method
    from the context.panda_testing_entries table"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('FailedEntriesBarPlotter: started', 'cyan')

        pf_failed_testing_entries = context.panda_failed_entries

        failed_method_counter = []
        for method in ComparisonMethods:
            failed_method_counter.append({
                'method': method.title,
                'count': pf_failed_testing_entries['failed_method'].str.count(method.name).sum()
            })

        pd_data = pd.DataFrame(failed_method_counter)
        print(pd_data)

        sns.set_theme()
        sns.set_context('paper')
        method_palette = [m.color for m in ComparisonMethods]

        sns.barplot(x='count',
                    y='method',
                    data=pd_data,
                    palette=method_palette,
                    orient='h')
        plt.title('Failed Entries')
        plt.show()

        cprint('FailedEntriesBarPlotter: done', 'green')
        next_step(context)
