#!/usr/bin/env python3
# coding: utf-8

import os
import time
from os.path import join
from typing import Dict, Any, Union

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import History


def path_prefix(path: str, identifier: str) -> str:
    """Returns a partial path for a given identifier.

    Arguments:
        path: An absolute path to an existing folder
        identifier: A runtime-unique indentifier
    Returns:
        A Partial path that is made complete by adding
        the name of the file, and its extension.
    Examples:
        path = os.path.abspath('some_existing_folder') \n
        id = get_identifier(path) \n
        partial_path = path_prefix(path, id) \n
        full_path = f'{partial_path}_array.npy' \n
            `full_path == '/abspath/some_existing_folder/id_array.npy'` \n
    """
    return join(path, identifier)


def get_identifier(path: str) -> str:
    """Generate an identifier unique to this runtime.
    The path argument is used to check for existing
    identifiers, so as to avoid duplicates.

    Arguments:
        path: An absolute path to an existing folder.
    Returns:
        A string of the format date-time-id, where the id
        is the current process id.
    Examples:
        path = os.path.abspath('some_existing_folder') \n
        id = get_identifier(path) \n
            `id == '2020-05-10-12-51-08-5432'` \n
    """

    t = time.strftime('%Y-%m-%d-%H-%M-%S')
    identifier = f'{t}-{os.getpid()}'
    if identifier in [_.split('_')[0] for _ in os.listdir(path)]:
        # Recursive call to generate a new ID
        time.sleep(2)
        return get_identifier(path)
    return identifier


def flatten(padded_sequences, decode=False) -> np.ndarray:
    # Shape of sequences is (num_sequences, padding_width, num_classes + 1)
    flat_encoded = np.array([
        one_hot_encoded for tokens
        in padded_sequences
        for one_hot_encoded in tokens
    ])
    # Shape of flat_encoded is (num_tokens, num_classes + 1)
    if decode:
        # one-hot to single integer representation
        flat_decoded = np.array([
            int(np.argmax(encoded))
            for encoded in flat_encoded
        ])
        # Shape of flat_encoded is (num_tokens,)
        return flat_decoded
    return flat_encoded


def y_dict(X: np.ndarray,
           y: np.ndarray,
           idx2lab: Dict[int, str]) -> Dict[str, Union[
    Dict[str, Union[np.ndarray, Any]],
    Dict[str, np.ndarray]]]:
    # Create an index array that "masks out" padding tokens
    pad_idx_arr = np.where(X.ravel() != 0.0)[0]

    # Create index arrays that maps out individual sentences
    sent_idx_arrays = [
        np.arange(len(sent)) for sent
        in [np.where(word != 0.0)[0] for word in X]
    ]

    vectorised_y = flatten(y, decode=True)
    vectorised_X = flatten(X, decode=True)

    unpadded = vectorised_y[pad_idx_arr]
    lab2idx = dict(zip(idx2lab.values(), idx2lab.keys()))
    majority_idx = lab2idx['O']
    binary = np.where(unpadded != majority_idx, 1, 0)
    readable = np.array([idx2lab[i] for i in unpadded])

    vec_sents = np.empty(shape=(len(X),), dtype=np.ndarray)
    bin_sents = vec_sents.copy()
    word_sents = vec_sents.copy()

    for i, ixs in enumerate(sent_idx_arrays):
        vec_sents[i] = unpadded[ixs]
        word_sents[i] = readable[ixs]
        bin_sents[i] = binary[ixs]
        vectorised_X = vectorised_X[len(ixs):]

    return {
        'flat': {
            'vectorised': unpadded,
            'binary': binary,
            'readable': readable
        },
        'sequential': {
            'vectorised': vec_sents,
            'binary': binary,
            'readable': word_sents
        }
    }


def plot_results(result: History, folder: str, metric: str, name: str) -> None:
    t = time.strftime('%m%d_%H-%M-%S')
    img_name = f'{t}_{name}_{metric}.png'
    path = join(folder, img_name)

    # Nicer x ticks
    x_ticks = np.arange(len(result.history[metric]), dtype=int)
    x_labels = x_ticks + 1

    plt.plot(result.history[metric], 'C2o-', label='training')
    if metric != 'lr':
        plt.plot(result.history[f'val_{metric}'], 'C1o--', label='validation')

    plt.title('BiLSTM (baseline model)')
    plt.ylabel(metric)
    plt.xlabel('epochs')
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.legend(title=metric.capitalize())

    plt.savefig(path)
    plt.close('all')
