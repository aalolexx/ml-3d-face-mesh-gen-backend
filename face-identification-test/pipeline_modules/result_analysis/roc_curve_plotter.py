from termcolor import cprint
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods



class RocCurvePlotter:
    """Plots the ROC Curve from the context.panda_testing_entries table"""
    def __init__(self, export_subdir: str, dataset_name: str) -> None:
        self._export_subdir = export_subdir
        self._dataset_name = dataset_name


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('RocCurvePlotter: started', 'cyan')

        pf = context.panda_testing_entries

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

        save_path = context.output_dir_path + '/' + self._export_subdir + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Plot the ROC curve
        sns.set_theme()
        sns.set_context('paper')
        fig, ax = plt.subplots()
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlim(-0.01, 1.01)
        plt.title('ROC Curve (using ' + self._dataset_name + ')')
        for cur_method_roc in roc_data:
            sns.lineplot(x=cur_method_roc['fpr'],
                         y=cur_method_roc['tpr'],
                         label=cur_method_roc['method_title'] + '(AUC = {:.3f})'.format(cur_method_roc['roc_auc']),
                         color=cur_method_roc['color'])

        plt.savefig(save_path + self._dataset_name + '_ROC_curve.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        cprint('RocCurvePlotter: done', 'green')
        next_step(context)