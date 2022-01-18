import pandas as pd
import numpy as np


def rps(y_true_one_hot: pd.DataFrame, prediction_proba: pd.DataFrame) -> float:
    """Computes RPS (Ranked Probability Score) given the one-hot encoded quintile
    dataframe and the predicted probability for each class.
    Note: both dataframes should have two-dimensional indexes with datetimes
    and asset IDs and one-dimensional columns with the quintile classes.

    :param y_true_one_hot: realised quintile (one-hot encoded format)
    :param prediction_proba: predicted probabilities per quintile
    :rtype: float
    """
    return (
        ((y_true_one_hot - prediction_proba) ** 2).mean(axis=1).unstack().mean(axis=1)
    )
