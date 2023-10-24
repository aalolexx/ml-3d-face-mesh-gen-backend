from termcolor import cprint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods

class ConfusionMatrixPlotter:
    """Plots the Confusion Matrix from the context.panda_dataframe table"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('ConfusionMatrixPlotter: started', 'cyan')

        pf = context.panda_dataframe

        pf["is_actual_match"] = pf["is_actual_match"].astype(int)

        pf_1 = pf[(pf.method == ComparisonMethods.COEFFICIENT_BASED_3D)]
        pf_2 = pf[(pf.method == ComparisonMethods.FACE_RECOGNITION_DISTANCE_2D)]

        confusion_1 = confusion_matrix(pf_1['is_actual_match'], pf_1['decision'])
        confusion_2 = confusion_matrix(pf_2['is_actual_match'], pf_2['decision'])

        # Create a heatmap to visualize the confusion matrix

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        sns.heatmap(confusion_1, annot=True, cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'], ax=ax1, cbar=False)
        sns.heatmap(confusion_2, annot=True, cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=[], ax=ax2, cbar=False)

        ax1.set_title('3D Coefficient Based')
        ax2.set_title('2D Face_Recognition')

        ax1.set(xlabel='Predicted', ylabel='Actual')
        ax2.set(xlabel='Predicted')

        ax1.set_aspect('equal', 'box')
        ax2.set_aspect('equal', 'box')

        plt.show()

        cprint('ConfusionMatrixPlotter: done', 'green')
        next_step(context)