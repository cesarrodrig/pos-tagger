"""
This script trains a model using a training and development dataset
provided as command-line arguments. For more details, see the function `cli`
or run `python train.py --help`.
"""

import os
from typing import Any, Dict, List

import click
import conllu
import dill  # noqa: F401 This package is joblib.dump works with lambdas
import joblib
from sklearn import base, pipeline
import yaml

import baseline
import feature_extraction
import hmm
import metrics

ModelConfig = Dict[str, Any]


def build_model_pipeline(model_config: ModelConfig) -> pipeline.Pipeline:
    """Construct the appropriate pipeline based on the model config.

    Parameters
    ----------
    model_config : ModelConfig
        Dict containing the information about the model. It must contain
        the keys `name` and `params`.

    Returns
    -------
    pipeline.Pipeline
        Built pipeline that can be used as a model to fit and predict.

    Raises
    ------
    TypeError
        If the model name passed is not recognized.
    """
    name = model_config['name']
    if name == 'baseline':
        return baseline.build_pipeline(model_config['params'])
    if name == 'hmm_nltk':
        return hmm.build_nltk_pipeline(model_config['params'])
    else:
        raise TypeError(f"Model `{name}` not recognized.")


def save_model(model: base.BaseEstimator,
               config: ModelConfig,
               acc: float,
               val_acc: float) -> None:
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
    model_name = config['name']
    # Name scheme: modelname__param1:val1__param2:val2.joblib
    params = [f"{p}:{v}" for p, v in config['params'].items()]
    params += [f"acc:{acc:0.04f}__val_acc:{val_acc:0.04f}"]
    model_name += "__" + "__".join(params)

    filename = os.path.join(config['save_dir'], f"{model_name}.joblib")
    joblib.dump(model, filename)
    print("Saved model in:", filename)


def train(train_data: List[conllu.TokenList],
          dev_data: List[conllu.TokenList],
          model_config: ModelConfig) -> None:
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

    model = build_model_pipeline(model_config)

    model.fit(train_data, y_train)

    y_pred = model.predict(train_data)
    accuracy_train = metrics.accuracy(y_train, y_pred)
    print("Model train accuracy:", accuracy_train)

    y_pred = model.predict(dev_data)
    accuracy_dev = metrics.accuracy(y_dev, y_pred)
    print("Model dev accuracy:", accuracy_dev)

    save_model(model, model_config, accuracy_train, accuracy_dev)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('train_filename')
@click.argument('dev_filename')
@click.option('--model',
              '-m',
              'model_filename',
              default='models/baseline.yaml',
              help="YAML file describing the model to use.")
def cli(train_filename: str, dev_filename: str, model_filename: str) -> None:
    """Train a model and evaluate its performance using a dev dataset.

    This script trains a model and evaluates its performance using
    the training and development datasets passed as arguments.
    """
    with open(train_filename, 'r') as f:
        train_data = conllu.parse(f.read())

    with open(dev_filename, 'r') as f:
        dev_data = conllu.parse(f.read())

    try:
        with open(model_filename, 'r') as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("No model definitation found, defaulting to `Baseline` model.")
        model_config = {'name': 'baseline'}

    train(train_data, dev_data, model_config)


if __name__ == '__main__':
    cli()
