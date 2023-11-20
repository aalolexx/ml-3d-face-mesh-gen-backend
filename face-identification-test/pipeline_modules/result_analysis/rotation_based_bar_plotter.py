from termcolor import cprint
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods

class RotationBasedBarPlotter:
    """Plots a Bar chart showing the accuracies by rotation angles
    from the context.panda_dataframe table"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('RotationBasedBarPlotter: started', 'cyan')

        pf = context.panda_dataframe

        rotation_method_grouped_pf = pf.groupby(['rotation_angle'])

        result_analysis = []

        for rotation_angle, group in rotation_method_grouped_pf:
            for method in ComparisonMethods:
                method_results = group[(group.method == method.name)]
                if method_results.shape[0] <= 0:
                    continue
                precision = precision_score(method_results['is_actual_match'], method_results['decision'])
                recall = recall_score(method_results['is_actual_match'], method_results['decision'])
                accuracy = accuracy_score(method_results['is_actual_match'], method_results['decision'])
                result_analysis.append({
                    'method': method.title,
                    'rotation': rotation_angle,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'count': method_results['open_testing_entry_id'].count()

                })

        seaborn_data = pd.DataFrame(result_analysis)

        sns.set_theme()
        sns.set_context('paper')
        method_palette = [m.color for m in ComparisonMethods]

        # Accuracy
        fig, ax = plt.subplots()
        ax.set_ylim(0.4, 1)
        sns.barplot(x='rotation', y='accuracy', hue='method', data=seaborn_data, palette=method_palette)
        sns.move_legend(ax, "lower right")
        plt.title('Accuracy')
        plt.show()

        # Precision
        fig, ax = plt.subplots()
        ax.set_ylim(0.4, 1)
        sns.barplot(x='rotation', y='precision', hue='method', data=seaborn_data, palette=method_palette)
        sns.move_legend(ax, "lower right")
        plt.title('Precision')
        plt.show()

        # Recall
        fig, ax = plt.subplots()
        ax.set_ylim(0.4, 1)
        sns.barplot(x='rotation', y='recall', hue='method', data=seaborn_data, palette=method_palette)
        sns.move_legend(ax, "lower right")
        plt.title('Recall')
        plt.show()

        sns.barplot(x='rotation', y='count', hue='method', data=seaborn_data, palette=method_palette)
        plt.show()

        cprint('RotationBasedBarPlotter: done', 'green')
        next_step(context)
