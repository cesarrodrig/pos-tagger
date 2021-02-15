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
    labels = np.zeros(len(sentences), dtype="object")
    for i, sentence in enumerate(sentences):
        labels[i] = [word["upos"] for word in sentence]

    return labels


class LemmaExtractor:
    """Class that extracts the lemmatized words from the sentences."""

    def fit(
        self, sentences: List[conllu.TokenList], y: np.array = None
    ) -> "LemmaExtractor":
        return self

    def transform(self, sentences: List[conllu.TokenList]) -> np.array:
        """Get the lemmatized version of a word and return it as the feature.

        Parameters
        ----------
        sentences : List[conllu.TokenList]
            List of tokenized sentences loaded with `conllu`.

        Returns
        -------
        np.array
            Array of lists of strings containing the words of the sentences.
        """
        features = np.zeros(len(sentences), dtype="object")
        for i, sentence in enumerate(sentences):
            features[i] = [word["lemma"] for word in sentence]

        return features
