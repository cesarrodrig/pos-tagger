"""
This module contains common-purpose classes and functions.
"""
import collections
from typing import Dict, List

import joblib
from sklearn import pipeline


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
