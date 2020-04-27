#!/bin/env python3
# coding: utf-8

import os
import pickle
from typing import Any, Optional, Union

import tensorflow as tf

from babil.data.preprocessing import Vocab
from babil.features.embeddings import WordEmbeddings


def pickle(obj: Union[Vocab, WordEmbeddings], target_dir: str):
    folder_ = os.path.abspath(target_dir)
    name = obj.__class__.__name__
    id_: str

    if isinstance(obj, WordEmbeddings):
        # Shave off the .zip/.txt/.whatever, and get the ID
        id_ = os.path.basename(obj.filepath).rsplit('.')[0]

    pickle_file = f'{name}_{id_}.pickle' if id_ else f'{name}.pickle'
    path_to_pickle = os.path.join(folder_, pickle_file)

    mode = 'wb' if os.path.exists(path_to_pickle) else 'xb'
    with open(path_to_pickle, mode) as f:
        pickle.dump(obj, f)
