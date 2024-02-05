from termcolor import cprint
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods
from pipeline_util.plot_util import *

class CategorizedBasedBarPlotter:
    """Plots a Bar chart showing the accuracies by category, eg. rotation angles
    from the context.panda_testing_entries table"""
    def __init__(self, group_by_category: str, export_subdir: str, dataset_name: str) -> None:
        self._group_by_category = group_by_category
        self._export_subdir = export_subdir
        self._dataset_name = dataset_name


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('CategorizedBasedBarPlotter: started', 'cyan')

        pf = context.panda_testing_entries

        category_method_grouped_pf = pf.groupby([self._group_by_category])

        result_analysis = []

        for cat, group in category_method_grouped_pf:
            for method in ComparisonMethods:
                method_results = group[(group.method == method.name)]
                if method_results.shape[0] <= 0:
                    continue
                precision = precision_score(method_results['is_actual_match'], method_results['decision'])
                recall = recall_score(method_results['is_actual_match'], method_results['decision'])
                accuracy = accuracy_score(method_results['is_actual_match'], method_results['decision'])
                result_analysis.append({
                    'method': method.title,
                    'category_name': cat,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'actual_mismatches': method_results.query('is_actual_match == 0')['open_testing_entry_id'].count(),
                    'actual_matches': method_results.query('is_actual_match == 1')['open_testing_entry_id'].count(),
                    'count': method_results['open_testing_entry_id'].count()
                })

        seaborn_data = pd.DataFrame(result_analysis)

        save_path = context.output_dir_path + '/' + self._export_subdir + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        set_sns_style(sns)
        method_palette = [m.color for m in ComparisonMethods]

        # Accuracy
        # Batplot
        fig, ax = plt.subplots()
        ax.set_ylim(0.4, 1)
        sns.barplot(x='category_name', y='Accuracy', hue='method', data=seaborn_data, palette=method_palette)
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False, fontsize='small')
        plt.title('Accuracy (using ' + self._dataset_name + ')')
        ax.set_xlabel(self._group_by_category)
        save_fig(plt, save_path + self._dataset_name + '_accuracy_bars_by' + self._group_by_category + '.png')
        plt.close()

        # Accuracy
        # Line Plot for Trend Line
        fig, ax = plt.subplots()
        ax.set_ylim(0.4, 1.01)
        sns.pointplot(x='category_name', y='Accuracy', hue='method', data=seaborn_data, palette=method_palette)
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False, fontsize='small')
        plt.title('Accuracy (using ' + self._dataset_name + ')')
        ax.set_xlabel(self._group_by_category)
        save_fig(plt, save_path + self._dataset_name + '_accuracy_lines_by' + self._group_by_category + '.png')
        plt.close()

        # Precision
        fig, ax = plt.subplots()
        ax.set_ylim(0.4, 1)
        sns.barplot(x='category_name', y='Precision', hue='method', data=seaborn_data, palette=method_palette)
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False, fontsize='small')
        plt.title('Precision (using ' + self._dataset_name + ')')
        # plt.show()
        ax.set_xlabel(self._group_by_category)
        save_fig(plt, save_path + self._dataset_name + '_precision_by' + self._group_by_category + '.png')
        plt.close()

        # Recall
        fig, ax = plt.subplots()
        ax.set_ylim(0.4, 1)
        sns.barplot(x='category_name', y='Recall', hue='method', data=seaborn_data, palette=method_palette)
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False, fontsize='small')
        plt.title('Recall (using ' + self._dataset_name + ')')
        # plt.show()
        ax.set_xlabel(self._group_by_category)
        save_fig(plt, save_path + self._dataset_name + '_recall_by' + self._group_by_category + '.png')
        plt.close()

        # Count Entries
        #pd.set_option('display.max_columns', None)
        #print(seaborn_data[['method', 'category_name', 'actual_mismatches', 'count']])

        #fig, ax = plt.subplots(1, 2)
        #sns.barplot(x='category_name', y='actual_matches', hue='method', data=seaborn_data, palette=method_palette, ax=ax[0])
        #sns.barplot(x='category_name', y='actual_mismatches', hue='method', data=seaborn_data, palette=method_palette, ax=ax[1])
        #plt.savefig(save_path + self._dataset_name + '_entry_count_by' + self._group_by_category + '.png',
        #            bbox_inches='tight', pad_inches=0)
        #plt.close()

        cprint('CategorizedBasedBarPlotter: done', 'green')
        next_step(context)
