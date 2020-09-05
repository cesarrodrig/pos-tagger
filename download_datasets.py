"""
This script download the training and development datasets so they can be
used to test the training script during CI.
"""

import os

import click
import wget


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('train_filename')
@click.argument('dev_filename')
def download(train_filename: str, dev_filename: str) -> None:

    train_url = """https://raw.githubusercontent.com/UniversalDependencies/\
UD_English-GUM/master/en_gum-ud-train.conllu"""
    dev_url = """https://raw.githubusercontent.com/UniversalDependencies/\
UD_English-GUM/master/en_gum-ud-dev.conllu"""

    if not os.path.isfile(train_filename):
        wget.download(train_url, train_filename)

    if not os.path.isfile(dev_filename):
        wget.download(dev_url, dev_filename)


if __name__ == "__main__":
    download()
