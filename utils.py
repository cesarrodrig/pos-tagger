"""
This module contains common-purpose classes and functions.
"""
import os
import collections
from typing import Dict, List

import joblib
from keras.preprocessing import sequence
import numpy as np
from sklearn import pipeline

import lstm


def load_model(filepath: str) -> pipeline.Pipeline:
    """Load a model from a file or directory.

    Parameters
    ----------
    filepath : str
        Path-like string.

    Returns
    -------
    pipeline.Pipeline
        Loaded model pipeline.
    """

    # REFACTOR: Very hacky and should be reworked into passing a model name.
    if os.path.isdir(filepath):
        return lstm.load_model_pipeline(filepath)

    return joblib.load(filepath)


class WordTagCounter:

    def __init__(self, default_tag: str = '') -> None:
        """Constructor

        Parameters
        ----------
        default_tag : str, optional
            Tag used when a word has not been seen before.
        """
        self.default_tag = default_tag

        # Store each tag count for every word the model is trained on.
        # I am not sure how much more efficient would be to have a fixed-sized
        # array containing the counts of each tag instead of Counter.
        # Using a dict and a counter is a little more intuitive and serves
        # the short-term purpose of building a baseline model.
        self._word_tag_counts = collections.defaultdict(
            lambda: collections.Counter()
        )  # type: Dict[str, collections.Counter]

    def update(self, words: List[str], tags: List[str]) -> None:
        """Update the counts of words to tags.


        Parameters
        ----------
        words : list
            List of strings.
        tags : list
            List of strings.
        """
        assert len(words) == len(tags), "X and y must be the same length."

        for word, tag in zip(words, tags):
            self._word_tag_counts[word].update([tag])

    def most_common_tag(self, word: str) -> str:
        """Return the most common tag associated with a word.

        If the word is not in the vocabulary, then return `default_tag`.

        Parameters
        ----------
        word : str
            Word to find.

        Returns
        -------
        str
            Most common tag.
        """
        counts = self._word_tag_counts[word]

        # If the word has not been seen, use the default tag
        if len(counts) == 0:
            return self.default_tag

        return counts.most_common(1)[0][0]

    def is_ambiguous(self, word: str) -> bool:
        """Determine whether a word has been seen with more than one tag.

        If the word has not been seen, it is assumed to be ambiguous.

        Parameters
        ----------
        word : str
            Word to find.

        Returns
        -------
        bool
            Whether the word has more than one tag.
        """
        if word not in self._word_tag_counts:
            return True

        return len(self._word_tag_counts[word]) > 1


class WordTokenizer:
    """Fits a vocabulary of words and maps them into indices as entries to a
    dictionary.

    The tokenizers provided in `sklearn` and `keras` all remove punctuation
    so this tokenizer solves this problem to provide us with a word to index
    mapping and its inverse index as well.
    """

    def __init__(self, unknown_token="") -> None:
        # Whenever an unseen token needs to be inversed, we return this.
        self.unknown_token = unknown_token

        # We keep mappings for word -> index and index -> word.
        self._index: Dict[str, int] = dict()
        self._inverse_index: Dict[int, str] = dict()

    def fit(self, words: List[str]) -> 'WordTokenizer':

        # Allows us to later to multi-index below.
        words = np.array(words)

        # idx contains the indices of unique words.
        # inv is an array of indices representing each word.
        self.vocabulary, idx, inv = np.unique(
            words, return_index=True, return_inverse=True)

        # We add two tokens, padding and out-of-vocab
        inv += 2
        oov_token = 1

        self._index = collections.defaultdict(lambda: oov_token,
                                              zip(words[idx], inv[idx]))
        self._inverse_index = collections.defaultdict(
            lambda: self.unknown_token, zip(inv[idx], words[idx]))

        return self

    def transform(self, words: List[str]) -> np.array:
        return np.array([self._index[w] for w in words])

    def inverse_transform(self, X: np.array) -> np.array:
        return np.array([self._inverse_index[i] for i in X])


class Sequencer:
    """Class that tokenizes sequences to categories and pads them to be
    the same length.
    """

    def __init__(self) -> None:
        self._word_tokenizer = WordTokenizer()

    def fit(self, sentences: List[List[str]]) -> 'Sequencer':
        words = [w for s in sentences for w in s]
        self._word_tokenizer.fit(words)

        # Save the length of the longest sentence for padding later.
        self._max_len = len(max(sentences, key=len))
        return self

    def transform(self, sentences: List[List[str]]) -> np.array:

        sentences = [self._word_tokenizer.transform(s) for s in sentences]
        sentences = sequence.pad_sequences(sentences,
                                           maxlen=self._max_len,
                                           padding='post')
        return sentences

    def inverse_transform(self, sequences: np.array) -> List[List[str]]:
        return [self._word_tokenizer.inverse_transform(s) for s in sequences]
