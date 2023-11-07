from termcolor import cprint
from sklearn.metrics import accuracy_score, precision_score
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

        print(pf.columns.tolist())

        rotation_method_grouped_pf = pf.groupby(['rotation_angle'])

        result_analysis = []

        for rotation_angle, group in rotation_method_grouped_pf:
            for method in ComparisonMethods:
                method_results = group[(group.method == method)]
                precision = precision_score(method_results['is_actual_match'], method_results['decision'])
                accuracy = accuracy_score(method_results['is_actual_match'], method_results['decision'])
                result_analysis.append({
                    'method': str(method),
                    'rotation': rotation_angle,
                    'accuracy': accuracy,
                    'precision': precision,
                    'count': len(method_results.index)

                })
                #print(str(rotation_angle) + ", " + str(method) + " -> " + str(accuracy) + " | " + str(precision))

        seaborn_data = pd.DataFrame(result_analysis)

        sns.set_theme()
        sns.set_context('paper')
        fig, ax = plt.subplots()
        ax.set_ylim(0.7, 1.1)
        sns.barplot(x='rotation', y='accuracy', hue='method', data=seaborn_data, palette=['seagreen', 'royalblue'])
        plt.show()

        fig, ax = plt.subplots()
        ax.set_ylim(0.2, 1.2)
        sns.barplot(x='rotation', y='precision', hue='method', data=seaborn_data, palette=['seagreen', 'royalblue'])
        plt.show()

        sns.barplot(x='rotation', y='count', hue='method', data=seaborn_data, palette=['seagreen', 'royalblue'])
        plt.show()

        cprint('RotationBasedBarPlotter: done', 'green')
        next_step(context)