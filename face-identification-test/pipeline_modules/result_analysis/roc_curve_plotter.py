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

        pf_1 = pf[(pf.method == ComparisonMethods.COEFFICIENT_BASED_3D)]
        pf_2 = pf[(pf.method == ComparisonMethods.FACE_RECOGNITION_DISTANCE_2D)]

        fpr_1, tpr_1, thresholds_1 = roc_curve(pf_1['is_actual_match'], pf_1['prediction'])
        fpr_2, tpr_2, thresholds_2 = roc_curve(pf_2['is_actual_match'], pf_2['prediction'])
        roc_auc_1 = roc_auc_score(pf_1['is_actual_match'], pf_1['prediction'])
        roc_auc_2 = roc_auc_score(pf_2['is_actual_match'], pf_2['prediction'])

        # Plot the ROC curve

        sns.set_theme()
        sns.set_context('paper')
        fig, ax = plt.subplots()
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.title('ROC Curve')
        sns.lineplot(x=fpr_1, y=tpr_1, label='ROC CB_3D (AUC = {:.2f})'.format(roc_auc_1), color='seagreen')
        sns.lineplot(x=fpr_2, y=tpr_2, label='ROC CB_2D (AUC = {:.2f})'.format(roc_auc_2), color='royalblue')
        plt.show()

        cprint('RocCurvePlotter: done', 'green')
        next_step(context)