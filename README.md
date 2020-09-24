# Part-Of-Speech Tagger

The exercise consists of building a Part-Of-Speech (PoS) tagger trained on the
[GUM corpus](https://corpling.uis.georgetown.edu/gum/) and using the
Universal Dependencies syntax annotations from
[here](https://github.com/UniversalDependencies/UD_English-GUM/tree/master).

## Instructions

To install the dependencies:

```
pip install -r requirements.txt
```

To download the datasets:

```
python download_datasets.py [train_filename] [dev_filename] [test_filename]
```

To train and save a model:

```
python train.py --model=[model_yaml] [train_filename] [dev_filename]
```

See `models/` for the three YAMLs containing the baseline, HMM, and LSTM
models.

The filename where the model is saved to is printed after finishing training.
Use these files in the evaluation step.

To evaluate an already trained model:

```
python eval.py [dev_filename] [model_filename]
```

To tag unlabeled sentences using a trained model:

```
python generate.py [text_filename] [model_filename]
```

To run all the previous steps for the three models implemented (except
LSTM training):

```
./build.sh
```

Training the LSTM takes around 20 minutes. To train it:

```
python train.py --model=models/lstm.yaml [train_filename] [dev_filename]
```

## Description

Part-Of-Speech tagging is the process of annotating the words of a sentence
with their grammatical role in a sentence with respect to its meaning
and context. Examples of tags include nouns, verbs, adjectives, and adverbs.

The main challenges of this problem are ambiguous words and unknown words.
Ambiguous words have more than one possible tag. For example, `transport`
can be a noun or a verb. Resolving ambiguous words can be done by using
statistical models and the context they appear in. Unknown words are items
that the model has not seen before, so it either has to predict the tag
or mark it an "unknown" tag.

Commonly-used models used as PoS taggers include:

* The baseline or naive model. Returns the most common tag of a word to
resolve ambiguities and it tags the word as a noun if it is unknown.
* Rule-based models. Solve ambiguous words by using handmade associations
between tags.
* Sequence models. Choose the sequence of tags based on its probability.
Examples include Hidden Markov Models (HMMs) and Recurrent Neural Networks
(RNNs).

## Dataset

The dataset we are using is the Universal Dependencies syntax annotations
of the GUM corpus. The files come in `CoNLL-U` format, composed of multiple
sentences and each sentence's words are represented with the following fields:

* `ID`: Word index.
* `FORM`: Word form or punctuation symbol.
* `LEMMA`: Lemma or stem of word form.
* `UPOS`: Universal part-of-speech tag.
* `XPOS`: Language-specific part-of-speech tag.
* `FEATS`: List of morphological features from the universal feature inventory.
* `HEAD`: Head of the current word.
* `DEPREL`: Universal dependency relation to the HEAD.
* `DEPS`: Enhanced dependency graph in the form of a list of head-deprel pairs.
* `MISC`: Any other annotation.

The goal is to train a model(s) to predict `UPOS` of each word in a sentence
using the other fields as features.

## Models

### Baseline Model

As a first step, we implement a baseline model so that we can compare other
models against it. This model works by counting the frequency that each tag
appears for each word and, during prediction, returns the most frequent tag.
If the word has not been seen before, it returns the noun tag.

This model is the simplesr solution and it can achieve an accuracy of 90%
[1](https://nlp.stanford.edu/fsnlp/) if trained with large datasets due to
getting common words like `a` and `the` right a lot.
Nevertheless, its performance is not great for unknown words at around 50%.

### Hidden Markov Models

HMMs are popular ways of building a PoS tagger. They choose the sequence of
tags with the maximum estimated likelihood based on previously seen sequences.

To make sure we cover a commonly used model, the HMM from NLTK was tested.

### Bidirectional LSTM

LSTMs are a kind of RNN that use gates to preserve state from previously seen
samples in a sequence. In a normal LSTM, this state is only passed from
first through last. In its bidirectional variation, there is another iteration
done when training on a sequence which is done from last to first. Doing this
enables the model to capture dependencies between samples that can happen
either before or after.

We use this model because we want to capture the meaning of words
according to the context they appear. The bidirectional LSTM models the
influence of surrounding words to and tries to infer the PoS tag this way.

### Metrics

We use three metrics to compare models:

* Accuracy. The percentage of tags that were correctly predicted. While this
metric is useful, it can be deceiving because the model may be predicting
trivial and frequent words correctly.

* Ambiguous words accuracy. The percentage of ambiguous words that were tagged
correctly. This metric reflects how well a model can infer the meaning of a
word that may have different ones.

* Unknown words accuracy. The percentage of words not seen during training
that were tagged correctly. This metric reflects how well a model can infer
the meaning of a word even without having seen it before.

## Assumptions

* The lemmatized version of the words is used since this minimizes
text ambiguity. In `generate.py`, the lemmatized version of the words are
obtained using the lemmatizer from `nltk`.

* The only feature used was the lemmatized words. The CoNLL-U format
has more features that could be included but tagging unlabeled sentences
would not be possible anymore.

* The unlabeled data is assumed to be spaced-separated tokens, this includes
punctuation like `.,?!`.

* Unseen words are considered ambiguous in the accuracy metric.

## Results

|       							| Baseline |  HMM   |  LSTM  |
|:----------------------------------|---------:|-------:|-------:|
| train accuracy        		   	| 93.31%   | 95.80% | 97.56% |
| train ambiguous words accuracy 	| 85.76%   | 91.08% | 95.52% |
| dev accuracy 						| 86.12%   | 77.20% | 89.72% |
| dev ambiguous words accuracy 		| 80.95%   | 67.22% | 87.36% |
| dev unknown words accuracy 		| 92.05%   | 84.48% | 93.90% |
| test accuracy 					| 85.04%   | 75.15% | 88.87% |
| test ambiguous words accuracy 	| 78.42%   | 73.91% | 87.01% |
| training time 					| 0.1723s  | 0.31s  | 1176s  |
| prediction time 					| 0.1966s  | 42.56s | 6.12s  |

### Discussion

The results from the baseline model served to corroborate that we are dealing
with a normal dataset since we expected an accuracy of around 90%. Given
that this dataset is not very large, it is more likely that unknown words
appear and were incorrectly labeled as a noun.

The results from the HMM were surprising since the literature states that it
has higher accuracy compared to the baseline. This poor performance is most
likely because of the small dataset. If it has not seen a sequence of tags
before, it will return a known sequence, even if it's wrong.

The results from the bidirectional LSTM were positive, improving over the
baseline by all metrics. The biggest gain was in the ambiguous words
accuracy. This is no surprise since we are taking advantage of both
the sequence modeling powers of LSTMs as well as the meaning inference
of word embeddings. Performance may even be improved if a hyperparameter
search was done.

### Conclusions

The baseline model helped us establish a performance that determines if a
model is useful or not as a PoS tagger. The HMM proved not useful by this
criteria while the LSTM proved to be a viable solution for this dataset.

The speed of the baseline model may be attractive to use it as a real-time
model and then use LSTM as a more accurate but slow background tagger.

## Potential Improvements

1. The baseline and HMM models produce the same results if trained with the
same data. RNNs have random initial states, so we are not guaranteed to obtain
the same results even with the same data. An improvement would be to do
several iterations per parameter combination and keep the best of each.

2. The hyperparameters of the LSTM were not optimized. Further work is needed
to find the optimal number of LSTM units and size of the word embeddings.

3. I am not 100% satisfied with how the LSTM model pipeline was implemented
and how it ended needing some taping around other classes. If I had more time,
I would finish encapsulating the logic so all models could be used by the
trainer seemlessly.

4. It would be heplful to implement an easier way to save and load models
from the scripts that doesn't have to know how a model is implemented.

5. I build `WordTagCounter` multiple times throughout the code based on the
training data. If the corpus gets large, this may both consume lots of memory
and take time to process. It would be worth it to have way of persisting it
and a good way of pass it around.

6. I did not have enough time to support calculating accuracy of unknown words
in `eval.py`. It would have required to save the `WordTagCounter` of the
training dataset and load it.

7. The directory structure needs some rework. Generally, there would be a
`src` folder containing the `.py` files and we keep files like
`requirements.txt` and `build.sh` in the root folder.
