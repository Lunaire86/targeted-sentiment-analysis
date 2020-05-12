#!/usr/bin/env python3
# coding: utf-8

from argparse import Namespace
from dataclasses import dataclass, field
from typing import List

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Embedding
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

METRICS = [
    CategoricalAccuracy(name='accuracy'),
    BinaryAccuracy(name='binary_accuracy'),
    TruePositives(name='tp'),
    TrueNegatives(name='tn'),
    FalsePositives(name='fp'),
    FalseNegatives(name='fn'),
    Precision(name='precision'),
    Recall(name='recall')
]


@dataclass
class Improved:
    # required args
    args: Namespace
    partial_path: str
    weights: np.ndarray

    # default value args
    num_classes: int = 5
    sequence_length: int = 50

    # fields set in __post_init__()
    path: str = field(init=False)
    callbacks: List[Callback] = field(init=False, default_factory=list)

    # class output
    model: Model = field(init=False, default=None)
    results: History = field(init=False, default=None)

    def __post_init__(self):

        self.callbacks = [
            EarlyStopping(
                monitor='fp',
                patience=5,
                verbose=1,
                mode='min'
            ),
            EarlyStopping(
                monitor='fn',
                patience=5,
                verbose=1,
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.1,
                patience=5,
                verbose=1
            ),
            # EarlyStopping(
            #     monitor='val_accuracy',
            #     min_delta=0.1,
            #     patience=5,
            #     verbose=2
            # ),
            ModelCheckpoint(
                filepath=f'{self.partial_path}_model_checkpoint_val_loss.h5',
                monitor='val_loss',
                save_format='h5',
                save_best_only=True,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-3,
                verbose=1
            )
        ]

    def build(self) -> None:

        """Build a model using the Keras functional API."""

        text_input = Input(
            shape=(self.sequence_length,),
            name='words'
        )
        embedded = Embedding(
            input_dim=self.weights.shape[0],  # vocab size
            output_dim=self.weights.shape[1],  # embedding dim
            input_length=self.sequence_length,  # num time-steps
            weights=[self.weights],
            mask_zero=True,
            trainable=self.args.train_embeddings,
            name='embeddings'
        )(text_input)

        # Fraction of the units to drop for
        # the linear transformation of the
        # -> inputs             (dropout)
        # -> recurrent state    (recurrent dropout)
        lstm_out = Bidirectional(
            LSTM(
                units=self.args.hidden_dim,
                dropout=self.args.dropout,
                recurrent_dropout=self.args.recurrent_dropout,
                return_sequences=True
            ),
            merge_mode='concat',
            name='BiLSTM-concat'
        )(embedded)

        predicted_labels = Dense(
            self.num_classes + 1,  # add one for padding token
            activation='softmax',
            name='output'
        )(lstm_out)

        self.model = Model(
            inputs=[text_input],
            outputs=[predicted_labels],
            name='improved'
        )
        optim = Adam(learning_rate=self.args.learning_rate)
        loss = CategoricalCrossentropy()

        self.model.compile(
            optimizer=optim,
            loss=loss,
            metrics=METRICS
        )
        self.model.summary()

    def summary(self):
        return self.model.summary()

    def train(self, X_train, y_train, X_dev, y_dev) -> None:
        """Train model."""
        self.results = self.model.fit(
            X_train, y_train,
            validation_data=(X_dev, y_dev),
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,  # samples per gradient
            shuffle=True,
            verbose=2,
            callbacks=self.callbacks,
            # ValueError: `class_weight` not supported for 3+ dimensional targets.
            # class_weight=class_weights
        )

    def save(self) -> None:
        """Save model."""
        self.model.save(f'{self.partial_path}_model.h5')
        # Save training metrics too
        metrics = np.array([
            self.results.history[_] for _ in self.results.history.keys()
        ])
        np.save(f'{self.partial_path}_metrics.npy', metrics)

    @staticmethod
    def load(path: str, checkpoint: bool = True, absolute_path: bool = False) -> Model:
        """Loads a pre-trained model. If path is not an absolute path
        pointing to a model saved in h5 format, the most recent model,
        is loaded, which by default is a checkpoint model.

        Arguments:
            path: Either an absolute path to a file, or a partial path
            that becomes absolute when appending a basename, which in this
            case is either `model_checkpoint.h5` or `model.h5`.
            checkpoint: Whether to load a checkpoint model or not.
            absolute_path: Toggle if path is an absolute path to a model.
        Returns:
            A :class:`~tensorflow.keras.models.Model` object.

        """
        name = 'model_checkpoint.h5' if checkpoint else 'model.h5'
        if absolute_path:
            model = keras.models.load_model(path)
        else:
            model = keras.models.load_model(f'{path}_{name}')

        return model

    def predict(self, *args, **kwargs):
        """Predict."""
        return self.model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Evaluate."""
        return self.model.evaluate(*args, **kwargs)
