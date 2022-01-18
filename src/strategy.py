import pandas as pd
import numpy as np


def get_weight_from_predictions(prediction: pd.DataFrame, k: float = 1) -> pd.DataFrame:
    """Given a prediction dataframe containing predicted probability of ranking,
    computes weights according to probability and ranking. The higher factor k,
    the more emphasis is put on the extremes of the distribution.

    :param prediction: dataframe with predicted ranking probability
    :param k: exponent of weight, defaults to 1. Needs to be an integer greater or equal to 1.
    :rtype: pd.DataFrame
    """

    if k < 1:
        raise ValueError("Parameter k needs to be >= 1")

    if not isinstance(k, int):
        k = int(k)

    winner_weight = (
        (
            prediction
            * (prediction.columns - np.median(prediction.columns)) ** (2 * k - 1)
        )
        .mean(axis=1)
        .unstack()
    )
    winner_weight = winner_weight.div(winner_weight.sum(axis=1), axis=0)
    return winner_weight
