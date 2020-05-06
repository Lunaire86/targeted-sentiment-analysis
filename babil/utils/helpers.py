#!/usr/bin/env python3
# coding: utf-8

import time
from os.path import join
from typing import Dict, Any, List, Union

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import History


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


def serialise(y_dict: Dict) -> Dict[Dict, List[Any]]:
    # TODO : this method is being a complete arse
    d = dict(y_dict)
    for k, v in d.items():
        for entry in v.keys():
            if isinstance(entry, np.ndarray):
                d[k][v][entry] = entry.tolist()
    return d


def y_dict(X: np.ndarray, y: np.ndarray,
           idx2lab: Dict[int, str],
           num_classes: int = 5) -> Dict[str, Union[Dict[str, Union[np.ndarray, Any]], Dict[str, np.ndarray]]]:
    # Create an index array that "masks out" padding tokens
    pad_idx_arr = np.where(X.ravel() != 0.0)[0]
    # Create index arrays that maps out individual sentences
    sent_idx_arrays = [
        np.arange(len(sent)) for sent
        in [np.where(word != 0.0)[0] for word in X]
    ]

    one_hot_y = flatten(y)
    vectorised_y = flatten(y, decode=True)
    vectorised_X = flatten(X, decode=True)

    unpadded = vectorised_y[pad_idx_arr]
    readable = np.array([idx2lab[i] for i in unpadded])

    n = num_classes + 1
    one_hot_sents = np.empty(shape=(len(X), n), dtype=np.ndarray)
    vec_sents = np.empty(shape=(len(X),), dtype=np.ndarray)
    word_sents = vec_sents.copy()

    for i, ixs in enumerate(sent_idx_arrays):
        vec_sents[i] = unpadded[ixs]
        word_sents[i] = readable[ixs]
        # Multiply sentence length by num_classes + 1
        # to get the correct shape and dim
        one_hot_sents[i] = one_hot_y[np.arange(ixs.size * n)]
        vectorised_X = vectorised_X[len(ixs):]

    return {
        'flat': {
            'one-hot': one_hot_y,
            'vectorised': unpadded,
            'readable': readable
        },
        'sequential': {
            'one-hot': one_hot_sents,
            'vectorised': vec_sents,
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
