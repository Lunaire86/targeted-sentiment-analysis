#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
import time
from argparse import Namespace
from logging import Logger
from os.path import join
from typing import Union

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.layers import Input, LSTM, LSTMCell, Masking
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from gensim.models import FastText
from gensim.models.fasttext import FastTextKeyedVectors, load_facebook_model, load_facebook_vectors

from data.preprocessing import Dataset, LabelTokeniser, WordTokeniser, vectorise
from utils.config import PathTracker


def plot_results(result: History, folder: str, metric: str) -> None:
    # TODO -- find out why this broke -- move to actual file ?
    t = time.strftime('%m%d_%H-%M-%S')
    img_name = f'{t}_baseline.png'
    path = join(folder, img_name)

    plt.plot(result.history[metric], 'C2', label='training')
    plt.plot(result.history[f'val_{metric}'], 'C1--', label='validation')
    plt.title('BiLSTM (baseline model)')
    plt.ylabel(metric)
    plt.xlabel('epochs')
    plt.legend(title=metric.capitalize())
    plt.show()
    plt.savefig(path, dpi=150)
    plt.close()


def mpl_setup():
    # sns.set()
    # sns.set_context('poster', font_scale=1.3)
    # sns.set_style("white")

    # Update matplotlib defaults to something nicer
    mpl_update = {
        'font.size': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.figsize': [12.0, 8.0],
        'axes.labelsize': 20,
        'axes.labelcolor': '#677385',
        'axes.titlesize': 20,
        'lines.color': '#0055A7',
        'lines.linewidth': 3,
        'text.color': '#677385',
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': 'Tahoma'
    }
    mpl.rcParams.update(mpl_update)
    # mpl.use('Qt5Agg')


def run(parsed_args: Namespace, paths: PathTracker, logger: Logger):
    # Based on checking the length of all sentences in train and dev,
    # setting max sequence length to 50 seems reasonable
    MAX_SEQUENCE_LENGTH = 50

    args = parsed_args
    path_to = paths
    mpl_setup()

    print('Loading the dataset...')
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)
    test = Dataset(path_to.test)  # Keep ur hands off, bro

    print('Loading pre-trained embeddings...')
    basename = 'cc.no.300.bin'
    path_to_embeddings = join(path_to.models, basename)
    embeddings: Union[FastText, FastTextKeyedVectors]

    if args.train_embeddings:
        embeddings = load_facebook_model(path_to_embeddings, encoding='latin1')
    else:
        embeddings = load_facebook_vectors(path_to_embeddings, encoding='latin1')

    # Get the weights and add vectors representing
    # indices 0 and 1 by concatenating along the first
    # axis (i.e. 'inserting' two rows at the front)
    embedding_dim = int(basename.split('.')[2])
    pad_vec = tf.random.uniform(shape=[1, embedding_dim], minval=0.05, maxval=0.95, seed=69686)
    unk_vec = tf.random.uniform(shape=[1, embedding_dim], minval=0.05, maxval=0.95, seed=69686)
    weights = np.r_[pad_vec.numpy(), unk_vec.numpy(), embeddings.vectors]

    # Update the vocab with tokens for padding and unknown words
    embeddings_vocab = embeddings.index2entity
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    word2idx.update(
        {word: idx + 2 for (idx, word) in enumerate(embeddings_vocab)}
    )
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Save the word-to-index dict
    w2i_path = join(path_to.models, f'{basename}_word2idx.pickle')
    mode = 'wb' if os.path.exists(w2i_path) else 'xb'
    with open(w2i_path, mode) as f:
        pickle.dump(word2idx, f)

    # And also save the weights using numpy
    npy_path = join(path_to.models, f'{basename}_weights.npy')
    mode = 'wb' if os.path.exists(npy_path) else 'xb'
    with open(npy_path, mode) as f:
        np.save(f, weights)

    print('Building the vocabulary...')
    # Words
    word_tokeniser = WordTokeniser()
    word_tokeniser.fit_on_texts(train.X)
    word_tokeniser.save(path_to.interim_data)
    print(f'Found {len(word_tokeniser.word_index)} different words.')

    # Labels
    label_tokeniser = LabelTokeniser()
    label_tokeniser.fit_on_texts(train.y)
    label_tokeniser.save(path_to.interim_data)
    num_classes = len(label_tokeniser.word_index)
    print(f'Found {len(label_tokeniser.word_index)} different labels.')

    # Vectorise words
    X_train = vectorise(train.X, word2idx=word2idx)
    X_dev = vectorise(dev.X, word2idx=word2idx)

    # Vectorise labels
    y_train = vectorise(train.y, label_tokeniser)
    y_dev = vectorise(dev.y, label_tokeniser)

    # TODO : include vocab from data (atm it's only from embeddings)

    # Build the model using the Keras functional API
    text_input = Input(
        shape=(MAX_SEQUENCE_LENGTH,),
        name='words'
    )
    embedded = Embedding(
        input_dim=weights.shape[0],        # vocab size
        output_dim=weights.shape[1],       # embedding dim
        input_length=MAX_SEQUENCE_LENGTH,
        weights=[weights],
        mask_zero=True,
        trainable=False,
        name='fastText'
    )(text_input)

    dropout = Dropout(
        rate=args.dropout,
        seed=69686,
        name='dropout'
    )(embedded)

    lstm_out = Bidirectional(
        LSTM(
            args.hidden_dim,
            recurrent_dropout=args.dropout,
            return_sequences=True
        ),
        name='BiLSTM'
    )(dropout)

    predicted_labels = Dense(
        num_classes + 1,  # add one for padding token
        activation='softmax',
        name='output'
    )(lstm_out)

    # TODO : check whether optim and loss can be called directly from compile()
    optim = Adam(learning_rate=args.learning_rate)     # alternatively, try Adamax
    loss = SparseCategoricalCrossentropy()

    model = Model(text_input, predicted_labels)
    model.compile(
        optimizer=optim,
        loss=loss,
        metrics=['accuracy'],
        name='baseline'
    )
    model.summary()

    model_path = join(path_to.models, 'baseline_checkpoint.h5')
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            verbose=2
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=2,
            verbose=2
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_format='h5',
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=1e-3,
            verbose=1
        )
    ]

    print('Training the model...')
    results = model.fit(
        X_train, y_train,
        validation_data=[X_dev, y_dev],
        epochs=args.epochs,
        batch_size=args.batch_size,  # samples per gradient
        verbose=2,
        shuffle=True,
        # callbacks=callbacks
    )

    model.save(join(path_to.models, 'baseline.h5'))

    plot_results(results, path_to.figures, 'accuracy')
    plot_results(results, path_to.figures, 'loss')



def dev_build():
    # Shrink stuff when running locally
    weights_ = weights[:20000]
    hidden_dim_ = 32

    # Build the model using the Keras functional API
    text_input = Input(
        shape=(MAX_SEQUENCE_LENGTH,),
        name='words'
    )
    embedded = Embedding(
        input_dim=weights_.shape[0],        # vocab size
        output_dim=weights_.shape[1],       # embedding dim
        input_length=MAX_SEQUENCE_LENGTH,
        weights=[weights_],
        mask_zero=True,
        trainable=False,
        name='fastText'
    )(text_input)

    dropout = Dropout(
        rate=args.dropout,
        seed=69686,
        name='dropout'
    )(embedded)

    lstm_out = Bidirectional(
        LSTM(
            hidden_dim_,
            recurrent_dropout=args.dropout,
            return_sequences=True
        ),
        name='BiLSTM'
    )(dropout)

    predicted_labels = Dense(
        num_classes + 1,  # add one for padding token
        activation='softmax',
        name='output'
    )(lstm_out)

    # TODO : check whether optim and loss can be called directly from compile()
    optim = Adam(learning_rate=args.learning_rate)     # alternatively, try Adamax
    loss = SparseCategoricalCrossentropy()

    model = Model(text_input, predicted_labels)
    model.compile(
        optimizer=optim,
        loss=loss,
        metrics=['accuracy'],
        name='baseline'
    )
    model.summary()
    return model


def dev_train(model: Model):
    epochs_ = 10
    model_path = join(path_to.models, 'baseline_checkpoint.h5')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=2,
            verbose=2
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_format='h5',
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            verbose=2
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=1e-3,
            verbose=1
        )
    ]

    print('Training the model...')
    results = model.fit(
        X_train, y_train,
        validation_data=[X_dev, y_dev],
        epochs=epochs_,
        batch_size=args.batch_size,  # samples per gradient
        verbose=2,
        shuffle=True,
        callbacks=callbacks
    )
    return results


if __name__ == '__main__':
    # Based on checking the length of all sentences in train and dev,
    # setting max sequence length to 50 seems reasonable
    MAX_SEQUENCE_LENGTH = 50
    MAX_VOCAB = 3000

    # Setup for running locally, as opposed to on Saga
    from babil.__main__ import dev_mode
    args, path_to, logger = dev_mode()
    mpl_setup()

    print('Loading the dataset...')
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)
    test = Dataset(path_to.test)  # Keep ur hands off, bro

    # We will just load the pickled files here
    # See run() for implementation of loading the actual model
    print('Loading pre-trained embeddings...')
    npy_path = join(path_to.models, 'cc.no.300.bin_weights.npy')
    w2i_path = join(path_to.models, 'cc.no.300.bin_word2idx.pickle')
    f = open(w2i_path, 'rb')

    weights = np.load(npy_path)
    word2idx = pickle.load(f)

    idx2word = {idx: word for (word, idx) in word2idx.items()}
    f.close()

    print('Building the vocabulary...')
    # Words
    word_tokeniser = WordTokeniser()
    word_tokeniser.fit_on_texts(train.X)
    word_tokeniser.save(path_to.interim_data)
    print(f'Found {len(word_tokeniser.word_index)} different words.')

    # Labels
    label_tokeniser = LabelTokeniser()
    label_tokeniser.fit_on_texts(train.y)
    label_tokeniser.save(path_to.interim_data)
    num_classes = len(label_tokeniser.word_index)
    print(f'Found {len(label_tokeniser.word_index)} different labels.')

    # Vectorise words
    X_train = vectorise(train.X, word2idx=word2idx)
    X_dev = vectorise(dev.X, word2idx=word2idx)

    # Vectorise labels
    y_train = vectorise(train.y, label_tokeniser)
    y_dev = vectorise(dev.y, label_tokeniser)
    # y_train = vectorise(train.y, label_tokeniser, categorical=True)
    # y_dev = vectorise(dev.y, label_tokeniser, categorical=True)

    # TODO : include vocab from data (atm it's only from embeddings)
    model = dev_build()
    # results = dev_train(model)

    # plot_results(results, path_to.figures, 'accuracy')
    # plot_results(results, path_to.figures, 'loss')
