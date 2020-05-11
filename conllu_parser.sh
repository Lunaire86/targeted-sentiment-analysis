#!/bin/bash

## Parse CoNNL-U files to cor, txt or other format.

find /cluster/work/users/marispau/nnc/ -name "avis-rest.s" -exec sh -c "cat {} | /cluster/work/users/marispau/in5550/sent_detagger.sh > {}.cor" \;


