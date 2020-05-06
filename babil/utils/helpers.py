#!/usr/bin/env python3
# coding: utf-8

import time
from os.path import join
from typing import Dict

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


def y_dict(X: np.ndarray, y: np.ndarray,
           idx2lab: Dict[int, str],
           num_classes: int = 5) -> Dict:
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

    one_hot_sents, vec_sents, word_sents = [], [], []
    n = num_classes + 1
    for ixs in sent_idx_arrays:
        vec_sents.append(unpadded[ixs])
        word_sents.append(readable[ixs])
        # Multiply sentence length by num_classes + 1
        one_hot_sents.append(one_hot_y[np.arange(ixs.size * n)])
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
