#!/bin/bash

# find . -iname '*.doc' -exec echo "File is {}" \;

# Usage: cat path_to.conllu | ./conllu_to_tsv.sh > path_to.tsv
#printf "form\n"

grep -v '^#' | awk -F'\t' -v form='' \
    '{ORS=" "} {if ($1 ~ /^$/) {printf "%s\n", form; form=""} else {form = form $2 " "}}' | sed 's/ \t/\t/g'

