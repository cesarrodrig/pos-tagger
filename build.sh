#!/bin/bash

set -e

flake8 *.py
mypy --ignore-missing-imports *.py
python download_datasets.py $TRAIN_FILENAME $DEV_FILENAME $TEST_FILENAME
echo "=== Training Baseline ==="
python train.py $TRAIN_FILENAME $DEV_FILENAME
echo "=== Evaluating Baseline ==="
python eval.py $TEST_FILENAME models/baseline__*.joblib
echo "=== Tagging with Baseline ==="
python generate.py $TEXT_FILENAME models/baseline__*.joblib > tags_baseline.txt
echo ""
echo "=== Training NLTK HMM ==="
python train.py $TRAIN_FILENAME $DEV_FILENAME --model models/hmm_nltk.yaml
echo "=== Evaluating NLTK HMM ==="
python eval.py $TEST_FILENAME models/hmm_nltk__*.joblib
echo "=== Tagging with NLTK HMM ==="
python generate.py $TEXT_FILENAME models/hmm_nltk__*.joblib > tags_hmm_nltk.txt
