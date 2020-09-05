"""
This script trains a model using a training and development dataset
provided as command-line arguments. For more details, see the function `cli`
or run `python train.py --help`.
"""

from typing import List

import click
import conllu
from sklearn import pipeline

import baseline
import feature_extraction
import metrics


def train(train_data: List[conllu.TokenList],
          dev_data: List[conllu.TokenList]) -> None:
    """Train the model and evaluate its performance using the dev dataset.

    Parameters
    ----------
    train_data : list
        List of sentences represented as `conllu.TokenList`.
    dev_data : list
        List of sentences represented as `conllu.TokenList`
    """
    print(f"Training with {len(train_data)} sentences.")
    print(f"Validating with {len(dev_data)} sentences.")

    y_train = feature_extraction.extract_tags(train_data)
    y_dev = feature_extraction.extract_tags(dev_data)

    model = pipeline.Pipeline([
        ('feature_extractor', baseline.FeatureExtractor()),
        ('model', baseline.Model()),
    ])

    model.fit(train_data, y_train)

    y_pred = model.predict(train_data)
    accuracy_train = metrics.accuracy(y_train, y_pred)
    print("Model train accuracy:", accuracy_train)

    y_pred = model.predict(dev_data)
    accuracy_dev = metrics.accuracy(y_dev, y_pred)
    print("Model dev accuracy:", accuracy_dev)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('train_filename')
@click.argument('dev_filename')
def cli(train_filename: str, dev_filename: str) -> None:
    """Train a model and evaluate its performance using a dev dataset.

    This script trains a model and evaluates its performance using
    the training and development datasets passed as arguments.
    """
    with open(train_filename, 'r') as f:
        train_data = conllu.parse(f.read())

    with open(dev_filename, 'r') as f:
        dev_data = conllu.parse(f.read())

    train(train_data, dev_data)


if __name__ == '__main__':
    cli()
