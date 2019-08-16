#!/usr/bin/env bash

set -e

STIMULI_FILE=../garden_path/data/verb-ambiguity-with-intervening-phrase.csv
VOCAB_RETAIN_SIZE=1000

# # First pre-compute corpus POS statistics.
# python get_corpus_statistics.py --pos VBD --outfile wiki_vbd.csv -n $VOCAB_RETAIN_SIZE

# Expand the stimulus set by replacing the disambiguating verb with many frequent verbs.
python expand_stimuli.py $STIMULI_FILE all_stimuli.csv

# Run the language model and get surprisals for each sentence.

# Index into each sentence and retrieve the critical surprisal.
