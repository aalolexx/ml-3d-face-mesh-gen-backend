from termcolor import cprint
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods


class RocCurvePlotter:
    """Plots the ROC Curve from the context.panda_dataframe table"""
    def __init__(self, additional_data_filter: str) -> None:
        self._additional_data_filter = additional_data_filter


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('RocCurvePlotter: started', 'cyan')

        pf = context.panda_dataframe
        if self._additional_data_filter:
            pf = pf[eval(self._additional_data_filter)]

        roc_data = []

        for method in ComparisonMethods:
            pf_cm = pf[(pf.method == method.name)]
            if pf_cm.shape[0] <= 0:
                continue

            fpr, tpr, thresholds = roc_curve(pf_cm['is_actual_match'], pf_cm['prediction'])
            roc_auc = roc_auc_score(pf_cm['is_actual_match'], pf_cm['prediction'])
            roc_data.append({
                'method_title': method.title,
                'color': method.color,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'roc_auc': roc_auc
            })

        # Plot the ROC curve
        sns.set_theme()
        sns.set_context('paper')
        sns.color_palette(['seagreen', 'royalblue', 'mediumorchid'])
        fig, ax = plt.subplots()
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.title('ROC Curve')
        for cur_method_roc in roc_data:
            sns.lineplot(x=cur_method_roc['fpr'],
                         y=cur_method_roc['tpr'],
                         label=cur_method_roc['method_title'] + '(AUC = {:.2f})'.format(cur_method_roc['roc_auc']),
                         color=cur_method_roc['color'])

        plt.show()

        cprint('RocCurvePlotter: done', 'green')
        next_step(context)