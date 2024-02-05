from termcolor import cprint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import colors
import pandas as pd
import os

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods
from pipeline_util.plot_util import *

class ConfusionMatrixPlotter:
    """Plots the Confusion Matrix from the context.panda_testing_entries table"""
    def __init__(self, export_subdir: str, dataset_name: str) -> None:
        self._export_subdir = export_subdir
        self._dataset_name = dataset_name


    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('ConfusionMatrixPlotter: started', 'cyan')

        pf = context.panda_testing_entries

        pf["is_actual_match"] = pf["is_actual_match"].astype(int)

        save_path = context.output_dir_path + '/' + self._export_subdir + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Prepare data for confusion matrices
        result_frames = []
        confusion_matrices = []

        for method in ComparisonMethods:
            current_pf = pf[(pf.method == method.name)]
            result_frames.append(current_pf)
            current_confusion = confusion_matrix(current_pf['is_actual_match'], current_pf['decision'])
            confusion_matrices.append(current_confusion)

        # Start Plotting
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 11))
        axes = axes.flatten()  # for iteration

        set_sns_style(sns)

        index = 0
        for method in ComparisonMethods:
            custom_cmap = self.create_custom_colormap('#1161b5')
            if '2d' in method.name:
                custom_cmap = self.create_custom_colormap('#b2298b')
            sns.heatmap(confusion_matrices[index],
                        annot=True,
                        fmt='g',
                        cmap=custom_cmap,
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'],
                        ax=axes[index],
                        cbar=False)
            axes[index].set_title(method.title)
            axes[index].set(xlabel='Predicted', ylabel='Actual')
            axes[index].set_aspect('equal', 'box')
            index += 1

        # Save Figure
        save_fig(plt, save_path + self._dataset_name + '_confusion_matrix.png')
        plt.close()

        cprint('ConfusionMatrixPlotter: done', 'green')
        next_step(context)

    # Creates a custom color gradient for the cmap param. Going from white to a given color
    def create_custom_colormap(self, hex_color):
        rgb_color = colors.to_rgb(hex_color)
        cmap_dict = {
            'red': [(0.0, 1.0, 1.0), (1.0, rgb_color[0], rgb_color[0])],
            'green': [(0.0, 1.0, 1.0), (1.0, rgb_color[1], rgb_color[1])],
            'blue': [(0.0, 1.0, 1.0), (1.0, rgb_color[2], rgb_color[2])],
        }
        custom_cmap = colors.LinearSegmentedColormap('custom_cmap', cmap_dict)
        return custom_cmap