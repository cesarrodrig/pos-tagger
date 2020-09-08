#!/bin/bash

set -e

flake8 *.py
mypy --ignore-missing-imports *.py
python download_datasets.py $TRAIN_FILENAME $DEV_FILENAME $TEST_FILENAME
python train.py $TRAIN_FILENAME $DEV_FILENAME
