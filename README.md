# Part-Of-Speech Tagger

The exercise consists of building a Part-Of-Speech (PoS) tagger trained on the
[GUM corpus](https://corpling.uis.georgetown.edu/gum/) and using the
Universal Dependencies syntax annotations from
[here](https://github.com/UniversalDependencies/UD_English-GUM/tree/master).

Part-Of-Speech tagging is the process of annotating the words of a sentence
with their grammatical role in a sentence with respect to its meaning
and context. Examples of tags include nouns, verbs, adjectives, and adverbs.

The main challenges of this problem are ambiguous words and unknown words.
Ambiguous words have more than one possible tag. For example, `transport`
can be a noun or a verb. Resolving ambiguous words can be done by using
statistical models and the context they appear in. Unknown words are items
that the model has not seen before, so it either has to predict the tag
or mark it an "unknown" tag.

## Models

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
