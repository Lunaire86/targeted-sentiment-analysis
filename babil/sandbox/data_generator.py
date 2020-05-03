#!/usr/bin/env python3
# coding: utf-8

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical
# from tensorflow.random import shuffle

from data.preprocessing import Data
from utils.config import set_global_seed


# @dataclass
# class DataGen(Sequence):
#     """Sequence is a safer way to do multiprocessing.
#     This structure guarantees that the network will
#     only train once on each sample per epoch which is
#     not the case with generators.
#     """
#     # X: Union[List, np.ndarray]
#     # y: Union[List, np.ndarray]
#     sequences: Data
#     batch_size: int = 32
#     # dim: Tuple[int] = (32, 32, 32)
#     # channels: int = 1
#     shuffle: bool = True
#     classes: int = field(init=False)
#     indices: List[int] = field(init=False, default_factory=list)
#
#     def __post_init__(self):
#         self.indices = np.indices(np.arange(len(self.sequences)))
#         self.classes = len(self.sequences.conll.get_labels())
#         set_global_seed()
#
#     def on_epoch_end(self):
#         """Updates the array of indices after each epoch."""
#         # TODO : find a better way of phrasing this
#         if self.shuffle:
#             self.indices = shuffle(self.indices, seed=42, name='end of epoch')
#
#     def __generate_batch(self, temp_sequences):
#         """Generates data containing [batch size] samples."""
#         # X = (num_samples, *dim, *num_channels) -- whatever that means, lol
#
#         # Find the length of the longest sequence
#         sequence_length = max(
#             max(train.as_lists().apply(len)),
#                               max(dev_tokens.apply(len))
#         )
#
#         # Initialisation
#         X = np.empty((self.batch_size, *self.dim, self.channels))
#         y = np.empty((self.batch_size), dtype=int)
#
#         # Generate data
#         for i, ID in enumerate(temp_sequences):
#             # Vectorise and pad sequences
#             X_ =
#             y_ = tf.data.Dataset.from_tensor_slices(
#                 list(map(lambda x: to_categorical(x), self.Data.y))
#             )
#         '''
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)
#
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')
#
#             # Store class
#             y[i] = self.labels[ID]
#
#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
#         '''
#
#     def __getitem__(self, index):
#         """Gets batch at position index"""
#         pass
#
#     def __len__(self):
#         """Number of batches in the Sequence (i.e. per epoch)."""
#         return int(np.floor(len(self.sequences) / self.batch_size))


@dataclass
class Minimal(Sequence):
    X: List[List[int]]
    y: List[List[int]]
    batch_size: int

    def __init__(self, X_data, y_data, batch_size):
        self.X, self.y = X_data, y_data
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        padded_X = np.array(pad_sequences(batch_X))
        padded_y = list(map(lambda x: to_categorical(x), pad_sequences(batch_y)))

        return padded_X, padded_y

    def __iter__(self):
        return super().__iter__()


if __name__ == '__main__':
    pass
