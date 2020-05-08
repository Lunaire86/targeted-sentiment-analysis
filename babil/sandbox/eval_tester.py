#!/usr/bin/env python3
# coding: utf-8
import os
from os.path import join
import time
from typing import List

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Bidirectional, Dense, Embedding
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import metrics as keras_metrics
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data.preprocessing import Dataset, WordTokeniser, LabelTokeniser, vectorise
from utils.config import set_global_seed, PathTracker
from utils.helpers import y_dict


def mpl_setup():
    sns.set()
    sns.set_context('poster', font_scale=1.1)
    sns.set_style("white")

    # Update matplotlib defaults to something nicer
    mpl_update = {
        'font.size': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': [12.0, 8.0],
        'figure.dpi': 150,
        'axes.labelsize': 16,
        'axes.labelcolor': '#677385',
        'axes.titlesize': 20,
        'lines.color': '#0055A7',
        'lines.linewidth': 2,
        'text.color': '#677385',
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': 'Tahoma'
    }
    mpl.rcParams.update(mpl_update)
    mpl.use('Agg')


def plot_results(result: History, folder: str, metric: str, model_name: str) -> None:
    t = time.strftime('%m%d_%H-%M-%S')
    img_name = f'{t}_{model_name}_{metric}.png'
    path = join(folder, img_name)

    # Nicer x ticks
    x_ticks = np.arange(len(result.history[metric]), dtype=int)
    x_labels = x_ticks + 1

    plt.plot(result.history[metric], 'C2o-', label='training')
    if metric != 'lr':
        plt.plot(result.history[f'val_{metric}'], 'C1o--', label='validation')
    plt.title(model_name)

    plt.ylabel(metric)
    plt.xlabel('epochs')
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.legend(title=metric.capitalize())
    plt.savefig(path)
    plt.close('all')


def _build(model_name: str) -> Model:
    # For this dev script, we're training our own word embeddings
    # Build the model using the Keras functional API
    text_input = Input(
        shape=(SEQUENCE_LENGTH,),
        name='words'
    )
    embedded = Embedding(
        input_dim=VOCAB_SIZE,  # vocab size
        output_dim=20,  # embedding dim
        input_length=SEQUENCE_LENGTH,
        mask_zero=True,
        trainable=True,
        name='self-trained'
    )(text_input)

    # Fraction of the units to drop for
    # the linear transformation of the
    # -> inputs             (dropout)
    # -> recurrent state    (recurrent dropout)
    lstm_out = Bidirectional(
        LSTM(
            units=4,  # hidden dim
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True
        ),
        name='Bi-LSTM'
    )(embedded)

    # num classes
    predicted_labels = Dense(
        num_classes + 1,  # +1 is for padding token when not using one-hot encoding
        activation='softmax',
        name='output'
    )(lstm_out)

    return Model([text_input], [predicted_labels], name=model_name)


def _compile(model: Model, metrics: List[Metric]) -> Model:
    optim = Adam(learning_rate=0.01)  # alternatively, try Adamax
    loss = CategoricalCrossentropy()

    model.compile(
        optimizer=optim,
        loss=loss,
        metrics=metrics,
    )
    model.summary()
    return model


def _fit(model: Model) -> History:
    checkpoint_path = join(path_to.models, 'mini_checkpoint.h5')
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            # min_delta=0.001,
            patience=3,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-3,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_format='h5',
            save_best_only=True,
        )
    ]

    print('Training the model...')
    results = model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=20,
        batch_size=32,  # samples per gradient
        shuffle=True,
        callbacks=callbacks
    )
    model.save(join(path_to.models, f'{model.name}.h5'))
    return results


def eval_on_test(filepath: str):
    # Load the test dataset
    test = Dataset(path_to.test)
    X_test = vectorise(test.X, word_tokeniser, maxlen=SEQUENCE_LENGTH)
    y_test = vectorise(test.y, label_tokeniser, maxlen=SEQUENCE_LENGTH)

    # Load the model we've trained
    model: Model = keras.models.load_model(filepath)
    pred = model.predict(X_test, batch_size=1, verbose=2)


def humanfriendlify(numerical_labels, label_tokeniser):
    # Convert gold labels from integers back to text:
    # Flattening labels from validation
    flattened = [y for x in numerical_labels for y in x]
    # Back to integers from one-hot
    integers = [int(np.argmax(pred)) for pred in flattened]
    # Back to text labels from integers
    humanfriendly = np.array([
        label_tokeniser.index_word[pred]
        if pred != 0
        else '<PAD>'
        for pred in integers
    ])
    return humanfriendly


if __name__ == '__main__':
    SEQUENCE_LENGTH = 30
    VOCAB_SIZE = 10000
    model: Model
    results: History

    mpl_setup()
    set_global_seed()
    print('Global seed set!')

    path_to = PathTracker.from_json(join(os.pardir, 'local_config.json'))

    print('Reading data from conll...')
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)
    print('Done!')

    print('Building the vocabulary...')
    # Words
    word_tokeniser = WordTokeniser(num_words=VOCAB_SIZE, lower=True)
    word_tokeniser.fit_on_texts(train.X)
    print(f'Found {len(word_tokeniser.word_index)} different words.')

    # Labels
    label_tokeniser = LabelTokeniser()
    label_tokeniser.fit_on_texts(train.y)
    num_classes = len(label_tokeniser.word_index)
    print(f'Found {len(label_tokeniser.word_index)} different labels.')

    # Vectorise words
    X_train = vectorise(train.X, word_tokeniser, maxlen=SEQUENCE_LENGTH)
    X_dev = vectorise(dev.X, word_tokeniser, maxlen=SEQUENCE_LENGTH)

    # Vectorise labels
    y_train = vectorise(train.y, label_tokeniser, maxlen=SEQUENCE_LENGTH, categorical=True)
    y_dev = vectorise(dev.y, label_tokeniser, maxlen=SEQUENCE_LENGTH, categorical=True)

    print(f'Train data shapes: X={X_train.shape}, y={y_train.shape}')   # X=(5915, 30) y=(5915, 30, 6)
    print(f'Dev data shapes: X={X_dev.shape}, y={y_dev.shape}')         # X=(1151, 30) y=(1151, 30, 6)

    # Metrics we'd like to use
    METRICS = [
        keras_metrics.CategoricalAccuracy(name='accuracy'),
        keras_metrics.BinaryAccuracy(name='binary_accuracy'),
        keras_metrics.TruePositives(name='tp'),
        keras_metrics.TrueNegatives(name='tn'),
        keras_metrics.FalsePositives(name='fp'),
        keras_metrics.FalseNegatives(name='fn'),
        keras_metrics.Precision(name='precision'),
        keras_metrics.Recall(name='recall'),
        # BinaryTruePositives(name='btp'),
        # BinaryTrueNegatives(name='btn'),
        # BinaryFalsePositives(name='bfp'),
        # BinaryFalseNegatives(name='bfn')
    ]

    __load__ = input('Load model? [Y] / N')
    if __load__.capitalize() != 'N':
        # Loading pre-trained model
        model = keras.models.load_model(join(path_to.models, 'mini_checkpoint.h5'))
        results = model.history
    else:
        # Building new model
        model = _build(model_name='mini')
        model = _compile(model, METRICS)
        results = _fit(model)

    # Get predictions
    predictions = model.predict(X_dev)

    y_pred = y_dict(X_dev, predictions, label_tokeniser.index_word)
    y_gold = y_dict(X_dev, y_dev, label_tokeniser.index_word)

    # print('Classification report for the dev set:')
    # print(classification_report(y_test_real, flattened_predictions))

    # plot_results(results, path_to.figures, 'binary_accuracy', model.name)
    # plot_results(results, path_to.figures, 'loss', model.name)
    # plot_results(results, path_to.figures, 'precision', model.name)
    # plot_results(results, path_to.figures, 'recall', model.name)
    # plot_results(results, path_to.figures, 'tp', model.name)
    # plot_results(results, path_to.figures, 'fp', model.name)
    # plot_results(results, path_to.figures, 'fn', model.name)
    # plot_results(results, path_to.figures, 'tn', model.name)
