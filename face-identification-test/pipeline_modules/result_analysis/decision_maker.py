from termcolor import cprint
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

from pipeline_modules.context import Context
from pipeline.pipeline import NextStep
from pipeline_util.enums import ComparisonMethods

class DecisionMaker:
    """Gets the optimal Threshold for each comparison method and makes an actual decision from the given prediction"""
    def __call__(self, context: Context, next_step: NextStep) -> None:
        cprint('------------------------------------', 'cyan')
        cprint('DecisionMaker: started', 'cyan')

        pf = context.panda_dataframe

        for method in ComparisonMethods:
            method_result_entries = pf[(pf.method == method)]
            # Get the optimal threshold using Youden J Method and ROC Curve
            fpr, tpr, thresholds = roc_curve(
                method_result_entries['is_actual_match'],
                method_result_entries['prediction']
            )
            youden_j = tpr - fpr
            optimal_threshold = thresholds[np.argmax(youden_j)]
            pf.loc[(pf.method == method), 'decision'] = np.where(
                method_result_entries['prediction'] > optimal_threshold,
                1,
                0
            )

        cprint('DecisionMaker: done', 'green')
        next_step(context)