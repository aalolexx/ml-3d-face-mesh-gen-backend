from termcolor import cprint
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods

class RocCurvePlotter:
    """Plots the ROC Curve from the context.panda_dataframe table"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('RocCurvePlotter: started', 'cyan')

        pf = context.panda_dataframe

        pf_1 = pf[(pf.method == ComparisonMethods.COEFFICIENT_BASED_3D)]
        pf_2 = pf[(pf.method == ComparisonMethods.FACE_RECOGNITION_DISTANCE_2D)]

        fpr_1, tpr_1, thresholds_1 = roc_curve(pf_1['is_actual_match'], pf_1['prediction'])
        fpr_2, tpr_2, thresholds_2 = roc_curve(pf_2['is_actual_match'], pf_2['prediction'])
        roc_auc_1 = roc_auc_score(pf_1['is_actual_match'], pf_1['prediction'])
        roc_auc_2 = roc_auc_score(pf_2['is_actual_match'], pf_2['prediction'])

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_1, tpr_1, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_1))
        plt.plot(fpr_2, tpr_2, color='green', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_2))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        cprint('RocCurvePlotter: done', 'green')
        next_step(context)