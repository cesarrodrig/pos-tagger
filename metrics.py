"""
This module contains the functions used to evaluate the performance of a
PoS tagger.
"""

import numpy as np
from sklearn import metrics


def accuracy(y_true: np.array, y_pred: np.array) -> float:
    """Calculate the percentage of correct tags.

    Parameters
    ----------
    y_true : np.array
        Correct labels.
    y_pred : np.array
        Predicted labels.
    """

    # sklearn expects 1D arrays so we flatten them.
    y_true = [word for sentence in y_true for word in sentence]
    y_pred = [word for sentence in y_pred for word in sentence]

    return metrics.accuracy_score(y_true, y_pred)
