"""
This module contains a PoS tagger that uses a bidirectional LSTM.
"""
import os
from typing import Any, Callable, Dict, List, Tuple

import conllu
import dill  # noqa: F401 This package is joblib.dump works with lambdas
import joblib
from keras import backend as K
from keras import callbacks, layers, models, optimizers
from keras import utils as keras_utils
import numpy as np

import feature_extraction
import utils


def build_pipeline(model_params: Dict[Any, Any]) -> "ModelPipeline":
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
    return ModelPipeline(model_params)


def load_model_pipeline(path: str) -> "ModelPipeline":
    """Load the model from a directory.

    The directory should contain the following files:

    * model.json
    * weights.h5
    * sentence_pipe.joblib
    * tag_pipe.joblib

    Parameters
    ----------
    path : str
        Directory where the model-related files are.

    Returns
    -------
    pipeline.Pipeline
        Loaded ModelPipeline
    """
    filename = os.path.join(path, "model.json")
    with open(filename, "r") as f:
        model = models.model_from_json(f.read())

    filename = os.path.join(path, "weights.h5")
    model.load_weights(filename)

    filename = os.path.join(path, "sentence_pipe.joblib")
    sentence_pipe = joblib.load(filename)

    filename = os.path.join(path, "tag_pipe.joblib")
    tag_pipe = joblib.load(filename)

    model_pipe = ModelPipeline()
    model_pipe.model = model
    model_pipe.sentence_pipe = sentence_pipe
    model_pipe.tag_pipe = tag_pipe

    return model_pipe


def build_model(
    seq_length: int,
    num_words: int,
    num_tags: int,
    units: int,
    embedding_units: int = 128,
) -> models.Model:
    """Build a Bidirectional LSTM that tags sequences using word embeddings.

    The architecture is as follows:

    The first layer is is a trainable word embeddings layer that we use to
    represent the meaning of a word with a vector.

    The second layer is a Bidirectional LSTM and it helps us model
    forward and backward dependencies between words.

    The third layer takes the outputs of the LSTM and passes them to a neural
    network to predict the probability of each tag for each word in a sentence.

    Parameters
    ----------
    seq_length : int
        Length of the sequences or sentences.
    num_words : int
        Size of the vocabulary.
    num_tags : int
        Number of different labels or tags.
    units : int
        LSTM units to use.
    embedding_units : int, optional
        Size of the word embeddings.

    Returns
    -------
    models.Model
        Keras Model.
    """
    inputs = layers.Input(shape=(seq_length,), dtype="int32")

    # +1 for out-of-vocabulary
    X = layers.Embedding(num_words + 1, embedding_units, trainable=True)(inputs)
    X = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(X)
    X = layers.TimeDistributed(layers.Dense(num_tags))(X)
    X = layers.Activation("softmax")(X)

    return models.Model(inputs=inputs, outputs=X)


class ModelPipeline:
    """Class that serves as the pipeline to use the Bidirectional LSTM
    as a sentence tagger.

    It was created to encapsulate the model from the feature extraction
    and transformation.
    """

    def __init__(self, params={}) -> None:
        """Constructor

        Parameters
        ----------
        params : dict, optional
            Model parameters containing `lstm_units`, `batch_size`,
            and `epochs`.
        """
        self.params = params
        self.sentence_pipe = SentencePipeline()
        self.tag_pipe = TagPipeline()

    def fit(
        self,
        sentences: List[conllu.TokenList],
        tags: List[List[str]],
        validation_data: Tuple[Any, Any] = None,
    ) -> None:
        """Fit the model.

        Parameters
        ----------
        sentences : list
            Training sentences.
        tags : List[str]
            Training tags of the sentences.
        validation_data : tuple, optional
            Validation sentences and tags.
        """
        X = self.sentence_pipe.fit(sentences).transform(sentences)
        y = self.tag_pipe.fit(tags).transform(tags, weighted=True)

        if validation_data:
            # The validation data also needs to be transformed.
            X_val = self.sentence_pipe.transform(validation_data[0])
            y_val = self.tag_pipe.transform(validation_data[1])
            validation_data = (X_val, y_val)

        # +2 to take into account pad and out-of-vocab.
        num_words = np.max(X) + 2
        num_tags = y.shape[-1]
        num_sentences, seq_len = X.shape

        lstm_units = self.params["lstm_units"]
        self.model = build_model(
            seq_length=seq_len, num_words=num_words, num_tags=num_tags, units=lstm_units
        )
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.Adam(0.01),
            metrics=["accuracy", non_pad_accuracy()],
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_ignore_pad_accuracy",
            min_delta=0,
            patience=5,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
        )

        batch_size = self.params["batch_size"]
        epochs = self.params["epochs"]
        self.model.fit(
            x=X,
            y=y,
            validation_data=validation_data,
            callbacks=[early_stopping],
            batch_size=batch_size,
            epochs=epochs,
        )

    def predict(self, sentences: List[conllu.TokenList]) -> List[List[str]]:
        """Label the sentences with their PoS tags.

        Parameters
        ----------
        sentences : list
            In the form of list of conllu.Tokens or list of strings.

        Returns
        -------
        list
            List of lists of strings with the predicted tags for each sentence.
        """
        X = self.sentence_pipe.transform(sentences)

        y_pred_probs = self.model.predict(X)

        # The caller expects tags, so we convert the probabilities all the
        # way to tag sequences.
        y_pred = np.argmax(y_pred_probs, axis=-1)
        y_pred = self.tag_pipe.sequencer.inverse_transform(y_pred)

        # There's probably a more elegant way of doing this.
        unpadded = []

        for sentence, tags in zip(sentences, y_pred):
            unpadded.append(tags[: len(sentence)])

        return unpadded

    def save(self, save_dir: str) -> None:
        """Save the model and its parts to a directory.

        The files saved in this function are needed by `load_model_pipeline`
        to correctly load this model pipeline.

        Parameters
        ----------
        save_dir : str
            Directory to save the model in.
        """
        # Make sure the directory exists
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        filename = os.path.join(save_dir, "model.json")
        model_json = self.model.to_json()
        with open(filename, "w") as f:
            f.write(model_json)

        filename = os.path.join(save_dir, "weights.h5")
        self.model.save_weights(filename)

        filename = os.path.join(save_dir, "tag_pipe.joblib")
        joblib.dump(self.tag_pipe, filename)

        filename = os.path.join(save_dir, "sentence_pipe.joblib")
        joblib.dump(self.sentence_pipe, filename)


class SentencePipeline:
    """Pipeline that takes lists of conllu.TokenList representing strings and
    convert them to integer sequences padded to the same length.
    """

    def __init__(self) -> None:
        self.lemma_extractor = feature_extraction.LemmaExtractor()
        self.sequencer = utils.Sequencer()

    def fit(self, sentences: List[conllu.TokenList]) -> "SentencePipeline":
        s = self.lemma_extractor.fit(sentences).transform(sentences)
        self.sequencer.fit(s)
        return self

    def transform(self, sentences: List[conllu.TokenList]) -> np.array:
        s = self.lemma_extractor.transform(sentences)
        return self.sequencer.transform(s)


class TagPipeline:
    """Pipeline that takes tags of sentences and converts them to one-hot
    encoding.
    """

    def __init__(self) -> None:
        self.sequencer = utils.Sequencer()

    def fit(self, tags: List[List[str]]) -> "TagPipeline":
        self.sequencer.fit(tags)
        return self

    def transform(self, tags: List[List[str]], weighted=False) -> np.array:
        sequences = self.sequencer.transform(tags)
        oh_sequences = keras_utils.to_categorical(sequences)

        if not weighted:
            return oh_sequences

        # The loss is cross entropy which is calculated as p*log(q) where p is
        # the true label and q the predicted probability of that label.
        # We can easily have weighted classes by modifying the target tags to
        # represent their weight instead of just being equal to 1.
        integer_sequences = np.argmax(oh_sequences, axis=2)
        bin_counts = np.bincount(integer_sequences.flatten())
        total = np.sum(bin_counts)
        for tag, count in enumerate(bin_counts):
            oh_sequences[:, :, tag] *= 1 - count / total

        # We have a lot of padding in the sequences, the model could give them
        # more weight at prediction time, so we adjust it to be very small.
        oh_sequences[:, :, 0] *= 0.00001

        return oh_sequences

    def inverse_transform(self, tags: np.array) -> List[List[str]]:
        # If we receive one-hot encoded tags, convert them to integers.
        if len(tags.shape) == 3:
            tags = np.argmax(tags, axis=2)

        # Now convert the integer form to the actual tags.
        return self.sequencer.inverse_transform(tags)


def non_pad_accuracy(pad_tag: int = 0) -> Callable:
    """Get function to calculate accuracy of tags that are not padding.

    Parameters
    ----------
    pad_tag : int, optional
        Padding class category.
    """

    def ignore_pad_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, pad_tag), "int32")
        matches = K.cast(K.equal(y_true_class, y_pred_class), "int32") * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_pad_accuracy
