"""
This script uses a previously trained model to predict the tags of an
unlabeled dataset. For more details, see the function `cli` or run
`python generate.py --help`.
"""

from typing import List

import click
import nltk
from nltk import stem
from sklearn import base

import utils

nltk.download('wordnet')


def generate(model: base.BaseEstimator, sentences: List[List[str]]) -> None:
    """Tag the sentences with the given model.

    Parameters
    ----------
    sentences : list
        List of lists of strings representing the sentences to tag.
    """
    print(f"Tagging {len(sentences)} sentences.")

    # Since the models were trained on the lemmatized version of the words,
    # we also lemmatize them when tagging unlabeled sentences.
    lemmatizer = stem.WordNetLemmatizer()

    for sentence in sentences:
        # Convert to the lemmatized versions
        lemmatized = [lemmatizer.lemmatize(w.lower()) for w in sentence]

        # Convert to conllu.TokenList because models expect that.
        # Since they are essentially dicts, we build them that way.
        tags = model.predict([[{'lemma': w} for w in lemmatized]])

        print("Word\tTag")
        for w, t in zip(sentence, tags[0]):
            print(f"{w}\t{t}")
        print()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('text_filename')
@click.argument('model_filename')
def cli(text_filename: str, model_filename: str) -> None:
    """Evaluates the performance of a model with the given dataset.
    """
    with open(text_filename, 'r') as f:
        unlabeled_data = f.readlines()

    model = utils.load_model(model_filename)

    # Convert to a list of lists of words.
    sentences: List[List[str]] = []
    for i in range(len(unlabeled_data)):
        sentence = [w for w in unlabeled_data[i].strip().split(' ')]
        sentences.append(sentence)

    generate(model, sentences)


if __name__ == '__main__':
    cli()
