"""
This module contains the functions used to evaluate the performance of a
PoS tagger.
"""

from typing import List

import conllu
import numpy as np
from sklearn import metrics

import utils


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


def ambiguous_accuracy(sentences: List[conllu.TokenList],
                       y_true: np.array,
                       y_pred: np.array) -> float:
    """Calculate the accuracy for words that have more than one tag.

    Parameters
    ----------
    sentences : list
        List of `conllu.TokenList` representing the sentences that were tagged.
    y_true : np.array
        Correct tags.
    y_pred : np.array
        Predicted tags.

    Returns
    -------
    float
        Accuracy score.
    """
    # Flatten them and convert to np for easier indexing.
    words = np.array(
        [word['lemma'] for sentence in sentences for word in sentence])
    y_true = np.array([tag for sentence in y_true for tag in sentence])
    y_pred = np.array([tag for sentence in y_pred for tag in sentence])

    counter = utils.WordTagCounter()
    counter.update(words, y_true)

    # Get the indices of the words that are ambiguous
    ambiguous_idx = [counter.is_ambiguous(word) for word in words]
    # Get the ambiguous true tags.
    ambiguous_true = y_true[ambiguous_idx]
    # Get the ambiguous predicted tags.
    ambiguous_pred = y_pred[ambiguous_idx]

    return metrics.accuracy_score(ambiguous_true, ambiguous_pred)


def unknown_accuracy(train_data: List[conllu.TokenList],
                     test_data: List[conllu.TokenList],
                     y_true: np.array,
                     y_pred: np.array) -> float:
    """Calculate the accuracy for words that are not in the training data.

    Parameters
    ----------
    train_data : list
        List of `conllu.TokenList` representing the sentences that were seen.
    test_data : list
        List of `conllu.TokenList` representing the sentences that were tagged.
    y_true : np.array
        Correct tags.
    y_pred : np.array
        Predicted tags.

    Returns
    -------
    float
        Accuracy score.
    """
    train_words = np.array(
        [word['lemma'] for sentence in train_data for word in sentence])
    test_words = np.array(
        [word['lemma'] for sentence in test_data for word in sentence])
    y_true = np.array([tag for sentence in y_true for tag in sentence])
    y_pred = np.array([tag for sentence in y_pred for tag in sentence])

    vocab = set(train_words)

    # Get the indices of the words that are ambiguous
    unknown_idx = [word in vocab for word in test_words]
    # Get the unknown true tags.
    unknown_true = y_true[unknown_idx]
    # Get the unknown predicted tags.
    unknown_pred = y_pred[unknown_idx]

    return metrics.accuracy_score(unknown_true, unknown_pred)
