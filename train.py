"""
This script trains a model using a training and development dataset
provided as command-line arguments. For more details, see the function `cli`
or run `python train.py --help`.
"""

from typing import List

import click
import conllu


def train(train_data: List[conllu.TokenList],
          dev_data: List[conllu.TokenList]) -> None:
    print(f"Training with {len(train_data)} sentences.")
    print(f"Validating with {len(dev_data)} sentences.")


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
