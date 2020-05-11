#!/bin/bash

## Parse CoNNL-U files to cor, txt or other format.

find . -name "*.conllu" -exec sh -c "cat {} | ./conllu_to_tsv.sh > {}.cor" \;


