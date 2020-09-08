"""
This script evaluates a model using a training and development dataset
provided as command-line arguments. For more details, see the function `cli`
or run `python train.py --help`.
"""

from typing import List

import click
import conllu
import joblib
from sklearn import base

import feature_extraction
import metrics


def eval(model: base.BaseEstimator, test_data: List[conllu.TokenList]) -> None:
    """Evaluate a model using the provided dataset.

    Parameters
    ----------
    test_data : list
        List of sentences represented as `conllu.TokenList`.
    """
    print(f"Evaluating with {len(test_data)} sentences.")

    y_test = feature_extraction.extract_tags(test_data)

    y_pred = model.predict(test_data)
    accuracy = metrics.accuracy(y_test, y_pred)
    print("Model accuracy:", accuracy)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('test_filename')
@click.argument('model_filename')
def cli(test_filename: str, model_filename: str) -> None:
    """Evaluates the performance of a model with the given dataset.
    """
    with open(test_filename, 'r') as f:
        test_data = conllu.parse(f.read())

    model = joblib.load(model_filename)
    eval(model, test_data)


if __name__ == '__main__':
    cli()