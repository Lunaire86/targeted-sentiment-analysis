#!/bin/env python3
# coding: utf-8

import os
import pickle
from typing import Any, Optional, Union

import tensorflow as tf

from src.data.preprocessing import Vocab
from src.features.embeddings import WordEmbeddings


def set_global_seed() -> Optional[Any]:
    # Ensure reproducibility
    global_seed: int

    with open('../SEED.txt', 'r') as f:
        global_seed = int(f.read())

    return tf.random.set_seed(global_seed)


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
