#!/usr/bin/env bash

set -e

# First pre-compute corpus POS statistics.
python get_items.py

# Expand the stimulus set by replacing the disambiguating verb with many frequent verbs.
python expand_stimuli.py

# Run the language model and get surprisals for each sentence.

# Index into each sentence and retrieve the critical surprisal.
