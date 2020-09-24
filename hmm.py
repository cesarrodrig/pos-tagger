"""
This module contains the classes used to implement PoS tagger using a HMM.
"""
from typing import Any, Dict, List

import conllu
import nltk
import numpy as np
from sklearn import base, pipeline


def build_nltk_pipeline(model_params: Dict[str, Any]) -> pipeline.Pipeline:
    """Build the HMM pipeline.

    Parameters
    ----------
    model_params : dict
        Model parameters.

    Returns
    -------
    pipeline.Pipeline
        Built pipeline that acts as the model.
    """
    return pipeline.Pipeline([
        ('feature_extractor', TokenExtractor()),
        ('model', NLTKModel(**model_params)),
    ])


class TokenExtractor:

    def fit(self,
            sentences: List[conllu.TokenList],
            y: np.array = None) -> 'TokenExtractor':
        """Noop.
        """
        return self

    def transform(self, sentences: List[conllu.TokenList]) -> List[List[str]]:
        """Extract the lemma words from the sentences coming from `conllu`.

        Parameters
        ----------
        sentences : list
            List of lists of tokens containing the sentences.

        Returns
        -------
        list
            List of lists of strings representing the sentences.
        """
        train_data = []
        for sentence in sentences:
            train_data.append([t['lemma'] for t in sentence])
        return train_data


class NLTKModel(base.BaseEstimator):

    def fit(self, X: List[List[str]], y: np.array) -> None:
        """Train the model.

        Parameters
        ----------
        X : list
            List of lists of strings containing the sentences.
        y : np.array, optional
            List of lists of strings containing the tags.
        """

        # The HMM offered by `nltk` requires the training data to be
        # a list of tuples with word and tag.
        train_data = []
        for i, sentence in enumerate(X):
            words_tags = [(w, t) for w, t in zip(sentence, y[i])]
            train_data.append(words_tags)

        trainer = nltk.tag.hmm.HiddenMarkovModelTrainer()
        self._model = trainer.train_supervised(train_data)

    def predict(self, X: List[List[str]]) -> List[List[str]]:
        """Summary

        Parameters
        ----------
        X : list
            List of lists of strings containing the sentences.

        Returns
        -------
        list
            List of lists of strings with the predicted tags.
        """
        return [self._model.best_path_simple(sentence) for sentence in X]
