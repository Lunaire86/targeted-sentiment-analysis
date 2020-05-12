#!/bin/bash

## Parse CoNNL-U files to cor, txt or other format.
# Copy the uncommented line below and run it in the terminal
# to pipe through a bunch of conll/conllu files in one go.

#find entry/point/ -name "*.conllu" -exec sh -c "cat {} | conllu_parser.sh > {}.cor" \;


grep -v '^#' | awk -F'\t' -v form='' \
    '{ORS=" "} {if ($1 ~ /^$/) {printf "%s\n", form; form=""} else {form = form $2 " "}}' | sed 's/ \t/\t/g'


