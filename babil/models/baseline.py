#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
from argparse import Namespace
from logging import Logger

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.layers import Input, LSTM, LSTMCell, Masking
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adamax, Adam

from babil.data.preprocessing import ConllData, Vocab
from babil.features.embeddings import load_embeddings
from babil.utils.config import ArgParser, PathTracker, set_global_seed
from babil.utils.helpers import save_pickle


def run(parsed_args: Namespace, paths: PathTracker, logger: Logger):
    args = parsed_args
    path_to = paths

    # Load datasets
    train = ConllData(path_to.train)
    dev = ConllData(path_to.dev)
    test = ConllData(path_to.test)  # Keep ur hands off, bro

    # Load embeddings
    print('Loading pre-trained embeddings...')
    embeddings = load_embeddings(args.embedding_id, path_to)
    small_embeddings = embeddings.__dummy__()
    print(f'Embeddings shape: {embeddings.shape}')
    print(f', dummy shape: {small_embeddings.shape}')

    # Create shared vocabulary for tasks
    print('Building the vocabulary...')
    vocab = Vocab()
    labels = train.get_labels()

    # Add words from both word embeddings and our training data
    vocab.add(embeddings.vocab)
    vocab.add(train.get_vocab())
    print('Done!')

    '''
    train_tokens = train_df['form_vec'].apply(np.ravel)
    train_labels = train_df['upos_vec'].apply(np.ravel)
    dev_tokens = dev_df['form_vec'].apply(np.ravel)
    dev_labels = dev_df['upos_vec'].apply(np.ravel)



    # Vectorise by mapping tokens to corresponding embedding
    # indices, then pad X_train and X_test to the same length
    X_train = pad_sequences(train_tokens, maxlen=sequence_length)
    y_train = pad_sequences(train_labels, maxlen=sequence_length)

    X_test = pad_sequences(dev_tokens, maxlen=sequence_length)
    y_test = pad_sequences(dev_labels, maxlen=sequence_length)
    '''
    # Find the length of the longest sequence
    # sequence_length = max(max(train.as_lists().apply(len)),
    #                       max(dev_tokens.apply(len)))

    # Pad sequences
    sequence_length = 50

    # Datasets get pickled automatically, but we
    # need to pickle vocab and embeddings manually
    print('Pickling embeddings and vocab...')
    save_pickle(embeddings, path_to.models)
    np.save(os.path.join(path_to.models, 'small-vec'), small_embeddings)
    save_pickle(vocab, path_to.data)
    print('Done!')
    # embeddings.pickle(os.path.join(root, 'models/embeddings/norwegian'))
    # vocab.pickle(os.path.join(root, 'data/interim'))

    # Create the embedding layer
    # embedding_layer = Embedding(
    #     input_dim=embeddings.vocab_size,     # vocab size
    #     output_dim=embeddings.dim,    # embedding dim
    #     input_length=sequence_length,
    #     weights=[embeddings.weights],
    #     mask_zero=True,
    #     trainable=False,
    # )

    # Create an Input layer
    # input_ = Input(shape=(sequence_length,), name='input')
    # x = embedding_layer(input_)
    # x = Dropout(0.3)(x)
    # x = Bidirectional(LSTM(args.hidden_dim, return_sequences=True))(x)
    # x = Bidirectional(LSTM(32))(x)
    # # x = LSTM(args.hidden_dim)(x)
    # x = Dense(64, activation='relu')(x)
    # output = Dense(len(labels), activation='softmax')(x)



if __name__ == '__main__':
    run()