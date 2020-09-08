"""
This module contains the classes used to implement a baseline PoS tagger.
"""

import collections
from typing import Any, Dict, List

import conllu
import numpy as np
from sklearn import base, pipeline


def build_pipeline(model_params: Dict[Any, Any]) -> pipeline.Pipeline:
    """Return a pipeline that can be used end-to-end with tokenized data.

    Parameters
    ----------
    model_params : dict
        Parameters that should be used to initialize the model.

    Returns
    -------
    pipeline.Pipeline
        Built pipeline that can be used as a model to fit and predict.
    """
    return pipeline.Pipeline([
        ('feature_extractor', FeatureExtractor()),
        ('model', Model(**model_params)),
    ])


class FeatureExtractor:
    """Class that converts tokenized sentences to the features needed by
    the baseline model.
    """

    def fit(self,
            sentences: List[conllu.TokenList],
            y: np.array = None) -> 'FeatureExtractor':
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
        features = np.zeros(len(sentences), dtype='object')
        for i, sentence in enumerate(sentences):
            features[i] = [word['lemma'] for word in sentence]

        return features


class Model(base.BaseEstimator):
    """Model that uses tag frequency to predict a words tag.

    This model is used as a baseline to evaluate other models against.
    It works by counting the frequency of each tag for each word in the
    training dataset and returning the most frequent tag as a prediction.
    """

    def __init__(self, default_tag: str = 'NOUN') -> None:
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

    def fit(self, X: np.array, y: np.array) -> None:
        """Train the model to fit the sentences to the tags.

        It iterates every word in every sentence, counting the number of times
        a tag is assigned to it.

        Parameters
        ----------
        X : np.array
            Array of lists of strings containing the words of the sentences.
        y : np.array
            Array of lists of strings containing the tags for the words.
        """
        assert len(X) == len(y), "X and y must be the same length."

        for i, sentence in enumerate(X):
            for j, word in enumerate(sentence):
                tag = y[i][j]
                self._word_tag_counts[word].update([tag])

    def predict(self, X: np.array) -> np.array:
        """Predict the tags of the words.

        The predicted tag is chosen to be the most frequent one or, if the
        word has not been seen before, it assigns it the `default_tag`.

        Parameters
        ----------
        X : np.array
            Array of lists of strings containing the words of the sentences
            which tags are to be predicted.

        Returns
        -------
        np.array
            Array of lists of strings containing the predicted tags.
        """
        pred = np.zeros(len(X), dtype='object')
        for i, sentence in enumerate(X):
            tags = []
            for word in sentence:
                counts = self._word_tag_counts[word]

                # If the word has not been seen, use the default tag
                if len(counts) == 0:
                    tags.append(self.default_tag)
                    continue

                # Pick the tag with the highest count.
                most_common = counts.most_common(1)[0][0]
                tags.append(most_common)

            pred[i] = tags

        return pred
