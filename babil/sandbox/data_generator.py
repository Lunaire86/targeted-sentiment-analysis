#!/usr/bin/env python3
# coding: utf-8

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.random import shuffle

from data.preprocessing import Dataset
from utils.config import set_global_seed


@dataclass
class DataGen(Sequence):
    """Sequence is a safer way to do multiprocessing.
    This structure guarantees that the network will
    only train once on each sample per epoch which is
    not the case with generators.
    """
    # X: Union[List, np.ndarray]
    # y: Union[List, np.ndarray]
    sequences: Dataset
    batch_size: int = 32
    dim: Tuple[int] = (32, 32, 32)
    channels: int = 1
    shuffle: bool = True
    classes: int = field(init=False)
    indices: List[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.indices = np.indices(np.arange(len(self.sequences)))
        self.classes = len(self.sequences.conll.get_labels())
        set_global_seed()

    def on_epoch_end(self):
        """Updates the array of indices after each epoch."""
        # TODO : find a better way of phrasing this
        if self.shuffle:
            self.indices = shuffle(self.indices, seed=42, name='end of epoch')

    def __generate_batch(self, temp_sequences):
        """Generates data containing [batch size] samples."""
        # X = (num_samples, *dim, *num_channels) -- whatever that means, lol

        # Find the length of the longest sequence
        sequence_length = max(
            max(train.as_lists().apply(len)),
                              max(dev_tokens.apply(len))
        )

        # Initialisation
        X = np.empty((self.batch_size, *self.dim, self.channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(temp_sequences):
            # Vectorise and pad sequences
            X_ = ''
            y_ = to_categorical()
        '''
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        '''

    def __getitem__(self, index):
        """Gets batch at position index"""
        pass

    def __len__(self):
        """Number of batches in the Sequence (i.e. per epoch)."""
        return int(np.floor(len(self.sequences) / self.batch_size))


if __name__ == '__main__':
    pass
