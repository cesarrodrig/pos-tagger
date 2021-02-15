"""
This script trains a model using a training and development dataset
provided as command-line arguments. For more details, see the function `cli`
or run `python train.py --help`.
"""

import os
import time
from typing import Any, Dict, List

import click
import conllu
import dill  # noqa: F401 This package is joblib.dump works with lambdas
import joblib
import numpy as np
from sklearn import base, pipeline
import yaml

import baseline
import feature_extraction
import hmm
import lstm
import metrics

ModelConfig = Dict[str, Any]

# These strings are ignored when creating a model filename.
IGNORE_PARAM_NAMES = {"weights_filename", "checkpoint_basepath"}


def build_and_fit(
    model_config: ModelConfig,
    x_train: np.array,
    y_train: np.array,
    x_val: np.array,
    y_val: np.array,
) -> pipeline.Pipeline:
    """Construct and fit a model with the given data.config.

    Parameters
    ----------
    model_config : ModelConfig
        Dict containing the information about the model. It must contain
        the keys `name` and `params`.
    x_train : np.array
        Training samples.
    y_train : np.array
        Training labels.
    x_val : np.array
        Validation samples.
    y_val : np.array
        Validation labels.

    Returns
    ------------------
    pipeline.Pipeline
        Pipeline with the fitted model.

    """
    name = model_config["name"]
    if name == "baseline":
        model = baseline.build_pipeline(model_config["params"])
        model.fit(x_train, y_train)

    elif name == "hmm_nltk":
        model = hmm.build_nltk_pipeline(model_config["params"])
        model.fit(x_train, y_train)

    elif name == "lstm":
        model = lstm.build_pipeline(model_config["params"])
        model.fit(x_train, y_train, validation_data=(x_val, y_val))

    else:
        raise TypeError(f"Model `{name}` not recognized.")

    return model


def save_model(
    model: base.BaseEstimator, config: ModelConfig, acc: float, val_acc: float
) -> None:
    """Persist a model using joblib.

    Parameters
    ----------
    model : base.BaseEstimator
        Model to save.
    config : ModelConfig
        Configuration used to build the model.
    acc : float
        Training accuracy.
    val_acc : float
        Validation accuracy.
    """
    model_name = config["name"]
    filtered_params = [
        (p, v) for p, v in config["params"].items() if p not in IGNORE_PARAM_NAMES
    ]

    # Name scheme: modelname__param1:val1__param2:val2.joblib
    params = [f"{p}:{v}" for p, v in filtered_params]
    params += [f"acc:{acc:0.04f}__val_acc:{val_acc:0.04f}"]
    model_name += "__" + "__".join(params)

    # Again, the LSTM model needs further rework so we don't have to be
    # taking care of these special cases. Probably a wrapper class Model
    # with a method 'save' that can be overloaded accordingly.
    if config["name"] == "lstm":
        # Saves the model architecture only.
        save_dir = os.path.join(config["save_dir"], model_name)
        model.save(save_dir)
        return

    filename = os.path.join(config["save_dir"], f"{model_name}.joblib")
    joblib.dump(model, filename)
    print("Saved model in:", filename)


def train(
    train_data: List[conllu.TokenList],
    dev_data: List[conllu.TokenList],
    model_config: ModelConfig,
) -> None:
    """Train the model and evaluate its performance using the dev dataset.

    Parameters
    ----------
    train_data : list
        List of sentences represented as `conllu.TokenList`.
    dev_data : list
        List of sentences represented as `conllu.TokenList`
    model_config : ModelConfig
        Dict containing the model information.
    """
    print(f"Training with {len(train_data)} sentences.")
    print(f"Validating with {len(dev_data)} sentences.")

    y_train = feature_extraction.extract_tags(train_data)
    y_dev = feature_extraction.extract_tags(dev_data)

    fit_start = time.time()

    model = build_and_fit(
        model_config=model_config,
        x_train=train_data,
        y_train=y_train,
        x_val=dev_data,
        y_val=y_dev,
    )

    fit_time = time.time() - fit_start

    pred_start = time.time()
    y_pred = model.predict(train_data)
    pred_time = time.time() - pred_start

    accuracy_train = metrics.accuracy(y_train, y_pred)
    amb_accuracy_train = metrics.ambiguous_accuracy(train_data, y_train, y_pred)

    y_pred = model.predict(dev_data)
    accuracy_dev = metrics.accuracy(y_dev, y_pred)
    amb_accuracy_dev = metrics.ambiguous_accuracy(dev_data, y_dev, y_pred)
    unk_accuracy_dev = metrics.unknown_accuracy(train_data, dev_data, y_dev, y_pred)

    print("Model train accuracy:", accuracy_train)
    print("Model dev accuracy:", accuracy_dev)

    print("Model train ambiguous words accuracy:", amb_accuracy_train)
    print("Model dev ambiguous words accuracy:", amb_accuracy_dev)

    print("Model unknown words accuracy:", unk_accuracy_dev)

    print(f"Training time: {fit_time:0.04f}s")
    print(f"Prediction time: {pred_time:0.04f}s")

    save_model(model, model_config, accuracy_train, accuracy_dev)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("train_filename")
@click.argument("dev_filename")
@click.option(
    "--model",
    "-m",
    "model_filename",
    default="models/baseline.yaml",
    help="YAML file describing the model to use.",
)
def cli(train_filename: str, dev_filename: str, model_filename: str) -> None:
    """Train a model and evaluate its performance using a dev dataset.

    This script trains a model and evaluates its performance using
    the training and development datasets passed as arguments.
    """
    with open(train_filename, "r") as f:
        train_data = conllu.parse(f.read())

    with open(dev_filename, "r") as f:
        dev_data = conllu.parse(f.read())

    try:
        with open(model_filename, "r") as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("No model definitation found, defaulting to `Baseline` model.")
        model_config = {"name": "baseline"}

    train(train_data, dev_data, model_config)


if __name__ == "__main__":
    cli()
