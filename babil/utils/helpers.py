#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
from typing import Union, Optional

from babil.data.preprocessing import Vocab
from babil.features.embeddings import WordEmbeddings


# Pickle Rick approves of these specific
# arguments because we can only allow for
# meaningful and relevant pickled objects.
APPROVED_BY_PICKLE_RICK = {
    'embeddings',
    'vocab',
    'train',
    'dev',
    'test'
}
# Wanna score some word embeddings?
# These are the paths you're looking for:
LOCATIONS = {
    'project_root': '/home/fero/Desktop/nlp/in5550-2020-exam',
    'local': '/media/fero/Programs/nlp/embeddings',
    'saga': '/cluster/shared/nlpl/vectors/latest',
    'project': '',
    'online': ''
} # TODO : ideally


def generate_path(arg: str, url: Optional[str] = '') -> str:
    base = ''
    if url:
        # TODO : Add support for ... something? TensorFlow Hub, GitHub, NLPL website?
        raise NotImplementedError('Coming in the near future. Maybe.')
    if arg in LOCATIONS:
        base = LOCATIONS[arg]
        return base
    raise ValueError(f'Valid arguments are {LOCATIONS}. '
                     f'If location=online, a valid url must be provided as the second argument.')


def open_pickle(flavour: str, file_id: Optional[str] = None):
    if flavour in APPROVED_BY_PICKLE_RICK:
        # TODO : implement
        return ''
    raise ValueError(f'Valid arguments are {APPROVED_BY_PICKLE_RICK}')



# def save_pickle(obj: Union[Vocab, WordEmbeddings], target_dir: str):
def save_pickle(obj, target_dir: str):
    folder_ = os.path.abspath(target_dir)
    name = obj.__class__.__name__
    id_ = ''

    if isinstance(obj, WordEmbeddings):
        # Shave off the .zip/.txt/.whatever, and get the ID
        id_ = os.path.basename(obj.filepath).rsplit('.')[0]

    pickle_file = f'{name}_{id_}.pickle' if id_ else f'{name}.pickle'
    path_to_pickle = os.path.join(folder_, pickle_file)

    mode = 'wb' if os.path.exists(path_to_pickle) else 'xb'
    with open(path_to_pickle, mode) as f:
        pickle.dump(obj, f)
