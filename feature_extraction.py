"""
This module contains functions used to extract features from the tokenized
sentences returned from `conllu`.
"""

from typing import List

import conllu
import numpy as np


def extract_tags(sentences: List[conllu.TokenList]) -> np.array:
    """Get the tags from the words of the sentences.

    Parameters
    ----------
    sentences : list
        List of tokenized sentences.

    Returns
    -------
    np.array
        Array of lists of strings containing the tags of the words.
    """
    labels = np.zeros(len(sentences), dtype='object')
    for i, sentence in enumerate(sentences):
        labels[i] = [word['upos'] for word in sentence]

    return labels
